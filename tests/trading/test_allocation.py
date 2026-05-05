from __future__ import annotations

import numpy as np
import pytest

import arena.tools.allocation as alloc


def _w_vec(res, tickers: list[str]) -> np.ndarray:
    return np.array([float(res.weights[t]) for t in tickers], dtype=float)


def test_sanitize_cov_makes_psd_and_symmetric() -> None:
    cov = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)  # eigenvalues: 3, -1
    cov[0, 0] = np.nan
    fixed = alloc.sanitize_cov(cov)
    assert fixed.shape == (2, 2)
    assert np.allclose(fixed, fixed.T, atol=1e-10)
    vals = np.linalg.eigvalsh(fixed)
    assert np.min(vals) >= -1e-12


def test_optimize_max_sharpe_outputs_valid_weights() -> None:
    rng = np.random.default_rng(0)
    tickers = ["A", "B", "C"]
    rets = rng.normal(loc=[0.001, 0.0005, 0.0002], scale=[0.02, 0.01, 0.015], size=(300, 3))
    res = alloc.optimize_max_sharpe(tickers, rets, risk_free_rate=0.02)
    w = _w_vec(res, tickers)
    assert set(res.weights.keys()) == set(tickers)
    assert np.all(w >= -1e-12)
    assert np.all(w <= 1.0 + 1e-12)
    assert abs(float(np.sum(w)) - 1.0) < 1e-6
    assert np.isfinite(res.sharpe)


def test_optimize_min_vol_prefers_lower_variance_asset() -> None:
    rng = np.random.default_rng(1)
    tickers = ["HIGH", "LOW"]
    high = rng.normal(loc=0.0005, scale=0.05, size=800)
    low = rng.normal(loc=0.0005, scale=0.01, size=800)
    rets = np.stack([high, low], axis=1)
    res = alloc.optimize_min_vol(tickers, rets)
    assert float(res.weights["LOW"]) > 0.8


def test_optimize_hrp_outputs_valid_weights() -> None:
    rng = np.random.default_rng(2)
    tickers = ["A", "B", "C", "D"]

    # two clusters: (A,B) and (C,D)
    f1 = rng.normal(0.0002, 0.01, size=600)
    f2 = rng.normal(0.0001, 0.012, size=600)
    a = f1 + rng.normal(0, 0.004, size=600)
    b = f1 + rng.normal(0, 0.004, size=600)
    c = f2 + rng.normal(0, 0.004, size=600)
    d = f2 + rng.normal(0, 0.004, size=600)

    rets = np.stack([a, b, c, d], axis=1)
    res = alloc.optimize_hrp(tickers, rets)
    w = _w_vec(res, tickers)
    assert abs(float(np.sum(w)) - 1.0) < 1e-6
    assert np.all(w >= -1e-12)


def test_optimize_blend_degenerates_to_components() -> None:
    rng = np.random.default_rng(3)
    tickers = ["A", "B", "C"]
    rets = rng.normal(loc=[0.001, 0.0004, 0.0003], scale=[0.02, 0.012, 0.015], size=(400, 3))

    ms = alloc.optimize_max_sharpe(tickers, rets)
    hrp = alloc.optimize_hrp(tickers, rets)

    blend_ms = alloc.optimize_blend(tickers, rets, sharpe_ratio=1.0, hrp_ratio=0.0)
    blend_hrp = alloc.optimize_blend(tickers, rets, sharpe_ratio=0.0, hrp_ratio=1.0)

    assert np.allclose(_w_vec(ms, tickers), _w_vec(blend_ms, tickers), atol=1e-8)
    assert np.allclose(_w_vec(hrp, tickers), _w_vec(blend_hrp, tickers), atol=1e-8)


def test_optimize_forecast_sharpe_responds_to_mu_when_scipy_available() -> None:
    if not getattr(alloc, "_HAS_SCIPY", False):
        pytest.skip("scipy not available")

    rng = np.random.default_rng(4)
    tickers = ["UP", "DOWN"]

    # identical historical distribution
    base = rng.normal(loc=0.0003, scale=0.02, size=700)
    rets = np.stack([base, base], axis=1)

    predicted_mu = {"UP": 0.25, "DOWN": -0.05}
    res = alloc.optimize_forecast_sharpe(tickers, rets, predicted_mu, mu_confidence=1.0)
    assert float(res.weights["UP"]) > float(res.weights["DOWN"])


def test_apply_weight_constraints_max_weight_redistributes_excess() -> None:
    w = {"A": 0.60, "B": 0.25, "C": 0.15}
    out = alloc.apply_weight_constraints(w, max_weight=0.35)
    assert out["A"] == pytest.approx(0.35, abs=1e-9)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert out["B"] > 0.25 and out["C"] > 0.15  # absorbed excess


def test_apply_weight_constraints_min_weight_drops_small_names() -> None:
    w = {"A": 0.50, "B": 0.30, "C": 0.18, "D": 0.02}
    out = alloc.apply_weight_constraints(w, min_weight=0.05)
    assert "D" not in out
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_apply_weight_constraints_cash_buffer_scales_sum() -> None:
    w = {"A": 0.6, "B": 0.4}
    out = alloc.apply_weight_constraints(w, cash_buffer=0.10)
    assert sum(out.values()) == pytest.approx(0.90, abs=1e-9)
    # proportions preserved
    assert out["A"] / out["B"] == pytest.approx(0.6 / 0.4, abs=1e-9)


def test_apply_weight_constraints_combined_order() -> None:
    w = {"A": 0.70, "B": 0.20, "C": 0.08, "D": 0.02}
    out = alloc.apply_weight_constraints(w, max_weight=0.40, min_weight=0.05, cash_buffer=0.10)
    assert "D" not in out
    assert max(out.values()) <= 0.40 * 0.90 + 1e-9
    assert sum(out.values()) == pytest.approx(0.90, abs=1e-9)
