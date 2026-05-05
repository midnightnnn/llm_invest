from __future__ import annotations

from arena.recommendation import signals as sg


def test_signal_catalog_exports_are_present() -> None:
    assert len(sg.ALL_SIGNALS) >= 15
    assert "momentum_20d" in sg.SIGNAL_NAMES
    assert "forecast_er" in sg.SIGNAL_NAMES
    assert "ep" in sg.SIGNAL_NAMES
    assert "low_debt" in sg.SIGNAL_NAMES


def test_signal_names_and_columns_match() -> None:
    for signal in sg.ALL_SIGNALS:
        assert signal.column == f"signal_{signal.name}"
        assert signal.direction in {"higher_better", "lower_better"}
        assert signal.group


def test_signal_column_lookup_is_unique() -> None:
    assert len(sg.SIGNAL_BY_COLUMN) == len(sg.ALL_SIGNALS)
    assert set(sg.SIGNAL_BY_COLUMN) == set(sg.SIGNAL_COLUMNS)


def test_signals_for_groups_filters() -> None:
    value_signals = sg.signals_for_groups("fundamental_value")
    assert {s.name for s in value_signals} == {"ep", "bp", "sp"}

    forecast_signals = sg.signals_for_groups("forecast")
    assert {s.name for s in forecast_signals} == {"forecast_er", "forecast_prob"}

    # Empty input → returns full catalog
    assert sg.signals_for_groups() == sg.ALL_SIGNALS


def test_regime_features_are_stable() -> None:
    assert "regime_vol_level" in sg.REGIME_FEATURES
    assert "regime_trend" in sg.REGIME_FEATURES
    assert "regime_dispersion" in sg.REGIME_FEATURES
    # Hard-coded order is relied on by the training code
    assert sg.REGIME_FEATURES[0] == "regime_vol_level"


def test_fundamentals_signals_cover_value_quality_growth_safety() -> None:
    groups = {s.group for s in sg.ALL_SIGNALS if s.group.startswith("fundamental_")}
    assert groups == {
        "fundamental_value",
        "fundamental_quality",
        "fundamental_growth",
        "fundamental_safety",
    }
