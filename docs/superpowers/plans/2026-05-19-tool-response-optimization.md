# Tool Response Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce batch-agent model-visible tool response tokens without losing investment decision information, by removing same-response duplication and making shared/default values explicit.

**Architecture:** Keep raw tool outputs unchanged. Optimize only the model-visible transcript in `arena/agents/adk_tool_compaction.py`, after raw tools return and before `attach_runtime_clock()`. Use lossless transforms: empty-container pruning, derived-count omission, mirror-list omission, and explicit `row_defaults` hoisting where every row shares a value.

**Tech Stack:** Python 3.11, pytest, DuckDB audit fixtures, ADK tool wrapper.

---

## File Structure

- Modify: `arena/agents/adk_tool_compaction.py`
  - Owns all model-visible response compaction.
  - Add small helper functions near existing `_compaction_meta()`.
  - Apply changes only inside known tool branches.
- Modify: `tests/adk/test_tool_compaction.py`
  - Add regression tests for each lossless optimization class.
  - Preserve existing tests that assert decision-critical fields remain visible.
- Optional inspect only: `arena/agents/adk_runner_bootstrap.py`
  - No planned code change. Confirms optimized payload is what the model sees.
- Optional inspect only: `arena/tools/quant_tools.py`, `arena/agents/adk_context_tools.py`, `arena/tools/sentiment_tools.py`
  - No planned code change unless a raw tool emits missing decision metadata that compaction currently hides.

## Lossless Rules

1. Do not remove raw tool fields at the tool implementation layer.
2. Do not remove values that the model cannot reconstruct from the same response.
3. If a per-row field is repeated across all rows and is decision-relevant, move it to `row_defaults` and remove it from each row.
4. If a top-level list mirrors `rows[].ticker` or `rows[].symbol`, omit the top-level list only when every visible row has the same ordered values and there are no excluded tickers hidden by the transform.
5. If `compaction.truncated` is `false`, omit `compaction.visible_count`, `returned_count`, and `visible_limit` when they are derivable from `rows`. Keep `compaction` when `truncated` is `true`.
6. Keep `errors`, `warnings`, `excluded`, `excluded_from_market_scope`, and freshness/degradation fields visible.
7. Do not merge semantically distinct aliases even when values match, for example `fear_greed_score` and `regime_score`.

---

### Task 1: Add Compaction Helper Tests

**Files:**
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add failing tests for generic lossless helpers**

Append these tests near the existing compaction tests:

```python
def test_compact_tool_result_technical_signals_omits_derived_meta_when_untruncated() -> None:
    out = _compact_tool_result_for_prompt(
        "technical_signals",
        {
            "tickers": ["AAPL", "MSFT"],
            "count": 2,
            "rows": [
                {
                    "ticker": "AAPL",
                    "price": 100.0,
                    "rsi_14": 61.2,
                    "rsi_state": "neutral",
                    "macd": {"state": "bullish"},
                    "moving_averages": {"price_vs_sma20": 0.02},
                    "bollinger_20_2": {"state": "inside_bands"},
                    "trend_state": "uptrend",
                },
                {
                    "ticker": "MSFT",
                    "price": 250.0,
                    "rsi_14": 48.0,
                    "rsi_state": "neutral",
                    "macd": {"state": "neutral"},
                    "moving_averages": {"price_vs_sma20": -0.01},
                    "bollinger_20_2": {"state": "inside_bands"},
                    "trend_state": "flat",
                },
            ],
        },
        args={"tickers": ["AAPL", "MSFT"]},
    )

    assert [row["ticker"] for row in out["rows"]] == ["AAPL", "MSFT"]
    assert "tickers" not in out
    assert "count" not in out
    assert "compaction" not in out
```

```python
def test_compact_tool_result_keeps_compaction_when_truncated() -> None:
    raw_rows = [
        {
            "ticker": f"T{i:02d}",
            "price": 100.0 + i,
            "rsi_14": 50.0,
            "rsi_state": "neutral",
            "macd": {"state": "neutral"},
            "moving_averages": {"price_vs_sma20": 0.0},
            "bollinger_20_2": {"state": "inside_bands"},
            "trend_state": "flat",
        }
        for i in range(11)
    ]

    out = _compact_tool_result_for_prompt(
        "technical_signals",
        {"tickers": [row["ticker"] for row in raw_rows], "count": 11, "rows": raw_rows},
        args={"tickers": [row["ticker"] for row in raw_rows]},
    )

    assert len(out["rows"]) == 10
    assert out["compaction"]["truncated"] is True
    assert out["compaction"]["returned_count"] == 11
    assert out["compaction"]["visible_count"] == 10
    assert out["compaction"]["visible_limit"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: the first new test fails because `tickers`, `count`, and untruncated `compaction` are still present.

---

### Task 2: Implement Generic Lossless Helpers

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Test: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add helper functions after `_compaction_meta()`**

```python
def _rows_count(value: Any, key: str = "rows") -> int | None:
    if isinstance(value, dict) and isinstance(value.get(key), list):
        return len(value.get(key) or [])
    return None


def _maybe_add_compaction(
    payload: dict[str, Any],
    *,
    requested_count: int | None,
    returned_count: int,
    visible_count: int,
    visible_limit: int,
) -> None:
    meta = _compaction_meta(
        requested_count=requested_count,
        returned_count=returned_count,
        visible_count=visible_count,
        visible_limit=visible_limit,
    )
    if meta["truncated"]:
        payload["compaction"] = meta
        return
    if requested_count is not None and requested_count != returned_count:
        payload["requested_count"] = requested_count


def _drop_if_row_field_mirror(
    payload: dict[str, Any],
    *,
    list_key: str,
    row_key: str,
    row_field: str,
) -> None:
    rows = payload.get(row_key)
    mirror = payload.get(list_key)
    if not isinstance(rows, list) or not isinstance(mirror, list):
        return
    values: list[Any] = []
    for row in rows:
        if not isinstance(row, dict) or row.get(row_field) is None:
            return
        values.append(row.get(row_field))
    if values == mirror:
        payload.pop(list_key, None)


def _drop_derived_count(payload: dict[str, Any], *, count_key: str = "count", row_key: str = "rows") -> None:
    row_count = _rows_count(payload, row_key)
    if row_count is not None and payload.get(count_key) == row_count:
        payload.pop(count_key, None)


def _drop_empty_dict(payload: dict[str, Any], key: str) -> None:
    if isinstance(payload.get(key), dict) and not payload.get(key):
        payload.pop(key, None)


def _hoist_constant_row_fields(
    payload: dict[str, Any],
    *,
    row_key: str,
    fields: tuple[str, ...],
    defaults_key: str = "row_defaults",
) -> None:
    rows = payload.get(row_key)
    if not isinstance(rows, list) or len(rows) < 2:
        return
    defaults: dict[str, Any] = {}
    for field in fields:
        values: list[Any] = []
        for row in rows:
            if not isinstance(row, dict) or field not in row:
                values = []
                break
            values.append(row.get(field))
        if values and all(value == values[0] for value in values[1:]):
            defaults[field] = values[0]
    if not defaults:
        return
    existing = payload.get(defaults_key)
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(defaults)
    payload[defaults_key] = merged
    for row in rows:
        if isinstance(row, dict):
            for field in defaults:
                row.pop(field, None)
```

- [ ] **Step 2: Apply helpers to `technical_signals` multi-row branch**

Replace the `compacted = {...}` block in the multi-row `technical_signals` branch with:

```python
            compacted = {
                "tickers": list(core.get("tickers") or [])[:visible_limit],
                "count": len(rows),
                "rows": rows,
            }
            excluded = list(core.get("excluded_from_market_scope") or [])
            if excluded:
                compacted["excluded_from_market_scope"] = excluded[:10]
            _maybe_add_compaction(
                compacted,
                requested_count=_requested_count(tool_args, core),
                returned_count=len(raw_rows),
                visible_count=len(rows),
                visible_limit=visible_limit,
            )
            if not excluded:
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="ticker")
            if "compaction" not in compacted:
                _drop_derived_count(compacted)
```

- [ ] **Step 3: Run tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: all tests pass, including existing truncation tests.

---

### Task 3: Optimize Event/Sentiment Multi-Row Metadata

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add tests for untruncated metadata omission**

```python
def test_compact_tool_result_earnings_calendar_omits_untruncated_compaction() -> None:
    out = _compact_tool_result_for_prompt(
        "earnings_calendar",
        {
            "ticker": None,
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2026-05-19",
            "days_ahead": 14,
            "count": 2,
            "rows": [
                {"date": "2026-05-20", "symbol": "AAPL", "name": "Apple", "time": "AMC", "eps_forecast": "1.60"},
                {"date": "2026-05-21", "symbol": "MSFT", "name": "Microsoft", "time": "BMO", "eps_forecast": "3.10"},
            ],
        },
        args={"tickers": ["AAPL", "MSFT"]},
    )

    assert "compaction" not in out
    assert "count" not in out
    assert "tickers" not in out
    assert [row["symbol"] for row in out["rows"]] == ["AAPL", "MSFT"]
```

```python
def test_compact_tool_result_reddit_keeps_requested_tickers_when_no_rows() -> None:
    out = _compact_tool_result_for_prompt(
        "fetch_reddit_sentiment",
        {"tickers": ["AAPL", "MSFT"], "count": 0, "rows": []},
        args={"tickers": ["AAPL", "MSFT"]},
    )

    assert out["tickers"] == ["AAPL", "MSFT"]
    assert out["rows"] == []
    assert "compaction" not in out
```

- [ ] **Step 2: Update `fetch_reddit_sentiment`, `fetch_sec_filings`, and `earnings_calendar` dict branches**

Use this pattern after each tool-specific `rows = _compact_rows(...)` call:

```python
            compacted = {
                "tickers": list(core.get("tickers") or []),
                "count": core.get("count", len(raw_rows)),
                "rows": rows,
            }
            _maybe_add_compaction(
                compacted,
                requested_count=_requested_count(tool_args, core),
                returned_count=len(raw_rows),
                visible_count=len(rows),
                visible_limit=visible_limit,
            )
            _drop_derived_count(compacted)
            if rows and "compaction" not in compacted:
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="ticker")
```

For `fetch_sec_filings`, keep:

```python
            if core.get("filing_type") is not None:
                compacted["filing_type"] = core.get("filing_type")
```

For `earnings_calendar`, keep:

```python
            compacted = {
                "ticker": core.get("ticker"),
                "start_date": core.get("start_date"),
                "days_ahead": core.get("days_ahead"),
                "count": core.get("count"),
                "rows": rows,
            }
            if core.get("tickers") is not None:
                compacted["tickers"] = core.get("tickers")
            _maybe_add_compaction(...)
            _drop_derived_count(compacted)
            if rows and "compaction" not in compacted:
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="symbol")
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py::test_compact_tool_result_earnings_calendar_omits_untruncated_compaction tests/adk/test_tool_compaction.py::test_compact_tool_result_reddit_keeps_requested_tickers_when_no_rows -q -p no:cacheprovider
```

Expected: both pass.

---

### Task 4: Optimize `recommend_opportunities`

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add tests for duplicate aliases and row defaults**

```python
def test_compact_recommend_opportunities_drops_empty_optimizer_and_confidence_alias() -> None:
    out = _compact_tool_result_for_prompt(
        "recommend_opportunities",
        {
            "status": "ok",
            "recommendations": [
                {
                    "ticker": "AAPL",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.81,
                    "score_source": "learned_ic",
                    "ranker_version": "v1",
                    "confidence": "medium",
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "reason_for": "strong trend",
                    "reason_risk": "valuation risk",
                },
                {
                    "ticker": "MSFT",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 2,
                    "recommendation_score": 0.72,
                    "score_source": "learned_ic",
                    "ranker_version": "v1",
                    "confidence": "medium",
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "reason_for": "quality momentum",
                    "reason_risk": "crowding risk",
                },
            ],
            "optimizer": {},
            "diagnostics": {"selection_scope": {"requested_profiles": ["balanced"], "effective_profiles": ["balanced"]}},
        },
    )

    assert "optimizer" not in out
    assert out["row_defaults"] == {
        "profile": "balanced",
        "bucket": "momentum",
        "score_source": "learned_ic",
        "ranker_version": "v1",
        "model_confidence": "medium",
        "action": "candidate",
        "evidence_level": "validated",
    }
    assert all("confidence" not in row for row in out["recommendations"])
    assert all("model_confidence" not in row for row in out["recommendations"])
    assert out["selection_scope"]["profiles"] == ["balanced"]
    assert "requested_profiles" not in out["selection_scope"]
    assert "effective_profiles" not in out["selection_scope"]
```

- [ ] **Step 2: Implement confidence alias collapse**

After building `compacted["recommendations"]`, add:

```python
        for row in compacted.get("recommendations") or []:
            if not isinstance(row, dict):
                continue
            if row.get("confidence") is not None and row.get("confidence") == row.get("model_confidence"):
                row.pop("confidence", None)
```

- [ ] **Step 3: Hoist constant recommendation fields**

Add after the alias collapse:

```python
        _hoist_constant_row_fields(
            compacted,
            row_key="recommendations",
            fields=(
                "profile",
                "bucket",
                "score_source",
                "ranker_version",
                "model_confidence",
                "action",
                "evidence_level",
            ),
        )
```

- [ ] **Step 4: Drop empty optimizer**

After the optimizer block:

```python
        _drop_empty_dict(compacted, "optimizer")
```

- [ ] **Step 5: Collapse identical requested/effective selection scope lists**

After building `compacted["selection_scope"]`, add:

```python
            selection_scope = compacted.get("selection_scope")
            if isinstance(selection_scope, dict):
                if selection_scope.get("requested_buckets") == selection_scope.get("effective_buckets"):
                    selection_scope["buckets"] = selection_scope.get("effective_buckets")
                    selection_scope.pop("requested_buckets", None)
                    selection_scope.pop("effective_buckets", None)
                if selection_scope.get("requested_profiles") == selection_scope.get("effective_profiles"):
                    selection_scope["profiles"] = selection_scope.get("effective_profiles")
                    selection_scope.pop("requested_profiles", None)
                    selection_scope.pop("effective_profiles", None)
```

- [ ] **Step 6: Run tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: all tests pass. Existing validation-field test must still prove `signal_contributions`, `optimizer_weight`, `score_components`, reasons, and recommendation scores remain visible.

---

### Task 5: Optimize `get_fundamentals`

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add test for derived counts**

```python
def test_compact_get_fundamentals_drops_derived_counts() -> None:
    out = _compact_tool_result_for_prompt(
        "get_fundamentals",
        {
            "requested": ["AAPL", "MSFT"],
            "eligible": ["AAPL", "MSFT"],
            "excluded": [],
            "rows": [
                {"ticker": "AAPL", "market": "us", "per": 31.5},
                {"ticker": "MSFT", "market": "us", "per": 34.0},
            ],
        },
    )

    assert "eligible_count" not in out
    assert "excluded_count" not in out
    assert out["requested_count"] == 2
    assert len(out["rows"]) == 2
```

- [ ] **Step 2: Drop counts only when derivable**

After the current `compacted = {...}` block:

```python
        if compacted.get("eligible_count") == len(rows):
            compacted.pop("eligible_count", None)
        if not excluded and compacted.get("excluded_count") == 0:
            compacted.pop("excluded_count", None)
        if excluded and compacted.get("excluded_count") == len(excluded[:5]):
            compacted.pop("excluded_count", None)
```

- [ ] **Step 3: Run tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py::test_compact_get_fundamentals_drops_derived_counts tests/adk/test_tool_compaction.py::test_compact_tool_result_get_fundamentals_reduces_meta_lists -q -p no:cacheprovider
```

Expected: both tests pass.

---

### Task 6: Optimize `portfolio_diagnosis`

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add test for benchmark alias collapse**

```python
def test_compact_portfolio_diagnosis_drops_duplicate_benchmark_alias() -> None:
    benchmark = {
        "ticker": "SPY",
        "return_krw": 0.05,
        "agent_return": -0.01,
        "excess_return_vs_benchmark": -0.06,
    }
    out = _compact_tool_result_for_prompt(
        "portfolio_diagnosis",
        {
            "risk_contribution": [{"ticker": "AAPL", "rc": 0.6}],
            "concentration_top3": 0.82,
            "hhi": 0.34,
            "momentum_20d_weighted": 0.07,
            "momentum_5d_weighted": 0.02,
            "volatility_20d_weighted": 0.18,
            "benchmark": benchmark,
            "benchmarks": {"current_sleeve": benchmark},
        },
    )

    assert "benchmark" not in out
    assert out["primary_benchmark_scope"] == "current_sleeve"
    assert out["benchmarks"]["current_sleeve"]["ticker"] == "SPY"
```

- [ ] **Step 2: Implement alias collapse**

After assigning `benchmarks`, replace the `benchmark` block with:

```python
        if core.get("benchmark") is not None:
            primary = _benchmark_for_prompt(core.get("benchmark"))
            benchmarks = compacted.get("benchmarks")
            matched_scope = None
            if isinstance(benchmarks, dict):
                for scope, benchmark in benchmarks.items():
                    if benchmark == primary:
                        matched_scope = str(scope)
                        break
            if matched_scope:
                compacted["primary_benchmark_scope"] = matched_scope
            else:
                compacted["benchmark"] = primary
```

- [ ] **Step 3: Run tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py::test_compact_portfolio_diagnosis_drops_duplicate_benchmark_alias tests/adk/test_context_tools_diagnosis.py::test_compact_portfolio_diagnosis_no_hrp_allocation -q -p no:cacheprovider
```

Expected: both pass.

---

### Task 7: Optimize `optimize_portfolio`

**Files:**
- Modify: `arena/agents/adk_tool_compaction.py`
- Modify: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Add test for target weight duplication**

```python
def test_compact_optimize_portfolio_uses_target_weights_map() -> None:
    out = _compact_tool_result_for_prompt(
        "optimize_portfolio",
        {
            "strategy": "sharpe",
            "expected_return_daily": 0.001,
            "volatility_daily": 0.02,
            "sharpe_daily": 0.4,
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "rebalance_orders": [
                {"ticker": "AAPL", "side": "BUY", "current_weight": 0.3, "target_weight": 0.6},
                {"ticker": "MSFT", "side": "SELL", "current_weight": 0.7, "target_weight": 0.4, "sell_ratio": 0.4286},
            ],
        },
    )

    assert out["target_weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert "allocations" not in out
    assert all("target_weight" not in order for order in out["rebalance_orders"])
    assert out["rebalance_orders"][0]["ticker"] == "AAPL"
```

- [ ] **Step 2: Replace allocation list with target weight map**

In the `optimize_portfolio` branch, replace `allocations` creation with:

```python
        weights = core.get("weights") or {}
        if isinstance(weights, dict):
            compacted["target_weights"] = {
                str(t): v
                for t, v in sorted(weights.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:12]
                if str(t).strip()
            }
```

- [ ] **Step 3: Remove order target weights when map has the same value**

After `orders = _compact_rows(...)`:

```python
        targets = compacted.get("target_weights") if isinstance(compacted.get("target_weights"), dict) else {}
        for order in orders:
            if isinstance(order, dict) and order.get("ticker") in targets and order.get("target_weight") == targets.get(order.get("ticker")):
                order.pop("target_weight", None)
```

- [ ] **Step 4: Run tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py::test_compact_optimize_portfolio_uses_target_weights_map -q -p no:cacheprovider
```

Expected: test passes and rebalance orders retain `ticker`, `side`, `current_weight`, and `sell_ratio`.

---

### Task 8: Audit All Remaining Tools

**Files:**
- Modify: `tests/adk/test_tool_compaction.py`
- Modify only if needed: `arena/agents/adk_tool_compaction.py`

- [ ] **Step 1: Add no-op guard tests for tools where matched values are semantically distinct**

```python
def test_compact_fear_greed_keeps_semantically_distinct_scores() -> None:
    out = _compact_tool_result_for_prompt(
        "fear_greed_index",
        {
            "fear_greed_score": 50.0,
            "regime_score": 50.0,
            "regime": "Neutral",
            "regime_label": "neutral",
            "source": "cboe_vix",
        },
    )

    assert out["fear_greed_score"] == 50.0
    assert out["regime_score"] == 50.0
    assert out["regime"] == "Neutral"
    assert out["regime_label"] == "neutral"
```

```python
def test_compact_macro_snapshot_shape_is_stable() -> None:
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "as_of": "2026-05-19",
            "source": "fred",
            "indicators": {
                "fed_funds_rate": {"value": 5.25, "date": "2026-05-01", "unit": "%"},
            },
        },
    )

    assert out == {
        "as_of": "2026-05-19",
        "source": "fred",
        "indicators": {
            "fed_funds_rate": {"value": 5.25, "date": "2026-05-01", "unit": "%"},
        },
    }
```

- [ ] **Step 2: Apply no code changes for these tools unless tests expose a real duplicate**

Tools expected to stay unchanged except for generic memory context behavior:

```text
fear_greed_index
macro_snapshot
index_snapshot
sector_summary
trade_performance
get_research_briefing
search_past_experiences
search_peer_lessons
scratch_run_python
save_memory
```

- [ ] **Step 3: Run full ADK compaction tests**

Run:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py tests/adk/test_runner_bootstrap.py -q -p no:cacheprovider
```

Expected: all pass.

---

### Task 9: Measure Token/Byte Savings Against Local Audit Data

**Files:**
- Create: `scripts/measure_tool_compaction.py`
- Test: manual command only

- [ ] **Step 1: Create measurement script**

```python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def main() -> None:
    db_path = Path("data/arena.duckdb")
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        """
        select tool_name, model_visible_result_json
        from agent_llm_tool_events
        where model_visible_result_json is not null
        """
    ).fetchall()
    sizes: dict[str, list[int]] = defaultdict(list)
    for tool_name, result_json in rows:
        sizes[str(tool_name)].append(_json_size(_parse(result_json)))
    for tool_name in sorted(sizes):
        values = sizes[tool_name]
        print(f"{tool_name:28s} n={len(values):4d} avg_bytes={sum(values) / len(values):8.1f} max_bytes={max(values):8d}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run baseline before code changes and optimized after code changes**

Run:

```bash
.venv/bin/python scripts/measure_tool_compaction.py
```

Expected: prints per-tool model-visible JSON sizes. Compare output before and after implementation by saving terminal output in the final implementation notes, not in the repo.

---

### Task 10: Final Verification

**Files:**
- Modified files from prior tasks

- [ ] **Step 1: Run focused tests**

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py tests/adk/test_runner_bootstrap.py tests/adk/test_context_tools_diagnosis.py -q -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 2: Run broader ADK tests**

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk -q -p no:cacheprovider
```

Expected: all pass or only unrelated known environment failures. Any failure touching tool compaction or runner bootstrap blocks completion.

- [ ] **Step 3: Review diff**

```bash
git diff -- arena/agents/adk_tool_compaction.py tests/adk/test_tool_compaction.py tests/adk/test_runner_bootstrap.py tests/adk/test_context_tools_diagnosis.py scripts/measure_tool_compaction.py
```

Expected: diff only changes model-visible compaction and tests. Raw tool implementations remain unchanged unless a missing decision-critical field required a passthrough.

---

## Self-Review

- Spec coverage: The plan covers information preservation, duplicate removal, token reduction, all default batch-agent tools, and audit measurement.
- Placeholder scan: No task uses "TBD", "TODO", or unspecified implementation.
- Type consistency: Helpers operate on `dict[str, Any]` and `list[dict[str, Any]]`, matching current compaction code.
- Risk note: `row_defaults` is explicit but changes response shape for affected dict-shaped tools. Top-level list-shaped tools are left stable unless a later measured patch justifies a shape migration.
