"""Layer 1 signal definitions for the opportunity ranker.

Each signal is a **deterministic** function of market/forecast/fundamentals data.
All signals share the same shape: one scalar per (as_of_date, ticker).

The ranker consumes these signals and learns the time-varying *information
coefficient* (IC) of each. The runtime score is a regime-conditional linear
combination: ``score = sum(predicted_IC_i * signal_i)``.

No signal depends on another signal. Correlated signals are allowed but
surface during IC audits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalDef:
    """Static metadata for a Layer 1 signal.

    The SQL expression references columns materialized by the signal refresh
    job inside ``signal_daily_values``. Expressions are kept simple so that
    IC/regime analytics can reason about monotone relationships.
    """

    name: str
    column: str
    direction: str
    group: str
    description: str


# Price / volatility signals derived from market_features
PRICE_SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        name="momentum_20d",
        column="signal_momentum_20d",
        direction="higher_better",
        group="price",
        description="Cross-section z-score of 20-day return (trend following).",
    ),
    SignalDef(
        name="pullback",
        column="signal_pullback",
        direction="higher_better",
        group="price",
        description="Positive 20d trend with short-term weakness (buy-the-dip).",
    ),
    SignalDef(
        name="meanrev_5d",
        column="signal_meanrev_5d",
        direction="higher_better",
        group="price",
        description="Negative of 5d z-score (short-term mean reversion).",
    ),
    SignalDef(
        name="lowvol",
        column="signal_lowvol",
        direction="higher_better",
        group="price",
        description="Negative of 20d volatility z-score (low-volatility anomaly).",
    ),
)

# Technical indicator signals — price-derived but non-linear
TECHNICAL_SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        name="rsi_reversal",
        column="signal_rsi_reversal",
        direction="higher_better",
        group="technical",
        description="Oversold/overbought classifier (RSI_14<30 → +1, >70 → -1).",
    ),
    SignalDef(
        name="ma_crossover",
        column="signal_ma_crossover",
        direction="higher_better",
        group="technical",
        description="SMA_20 vs SMA_60 regime (+1 bull cross, -1 bear cross).",
    ),
    SignalDef(
        name="bollinger_position",
        column="signal_bollinger_position",
        direction="higher_better",
        group="technical",
        description="Normalized distance from SMA_20 scaled by 2×std_20.",
    ),
)

# Sentiment signal
SENTIMENT_SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        name="sentiment",
        column="signal_sentiment",
        direction="higher_better",
        group="sentiment",
        description="Cross-section z-score of sentiment_score.",
    ),
)

# Forecast signals — consumed output of neuralforecast ML
FORECAST_SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        name="forecast_er",
        column="signal_forecast_er",
        direction="higher_better",
        group="forecast",
        description="Stacked neuralforecast expected return over horizon.",
    ),
    SignalDef(
        name="forecast_prob",
        column="signal_forecast_prob",
        direction="higher_better",
        group="forecast",
        description="Stacked forecast probability-of-up minus 0.5 (confidence delta).",
    ),
)

# Fundamentals signals — point-in-time derived ratios
FUNDAMENTAL_SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        name="ep",
        column="signal_ep",
        direction="higher_better",
        group="fundamental_value",
        description="Earnings yield (EPS_TTM / price) z-score.",
    ),
    SignalDef(
        name="bp",
        column="signal_bp",
        direction="higher_better",
        group="fundamental_value",
        description="Book-to-price z-score.",
    ),
    SignalDef(
        name="sp",
        column="signal_sp",
        direction="higher_better",
        group="fundamental_value",
        description="Sales-to-price z-score.",
    ),
    SignalDef(
        name="roe",
        column="signal_roe",
        direction="higher_better",
        group="fundamental_quality",
        description="Return on equity z-score.",
    ),
    SignalDef(
        name="revenue_growth",
        column="signal_revenue_growth",
        direction="higher_better",
        group="fundamental_growth",
        description="Year-over-year revenue growth z-score.",
    ),
    SignalDef(
        name="eps_growth",
        column="signal_eps_growth",
        direction="higher_better",
        group="fundamental_growth",
        description="Year-over-year EPS growth z-score.",
    ),
    SignalDef(
        name="low_debt",
        column="signal_low_debt",
        direction="higher_better",
        group="fundamental_safety",
        description="Negative of debt-to-equity z-score.",
    ),
)


ALL_SIGNALS: tuple[SignalDef, ...] = (
    *PRICE_SIGNALS,
    *TECHNICAL_SIGNALS,
    *SENTIMENT_SIGNALS,
    *FORECAST_SIGNALS,
    *FUNDAMENTAL_SIGNALS,
)

SIGNAL_NAMES: tuple[str, ...] = tuple(s.name for s in ALL_SIGNALS)
SIGNAL_COLUMNS: tuple[str, ...] = tuple(s.column for s in ALL_SIGNALS)
SIGNAL_BY_NAME: dict[str, SignalDef] = {s.name: s for s in ALL_SIGNALS}
SIGNAL_BY_COLUMN: dict[str, SignalDef] = {s.column: s for s in ALL_SIGNALS}


# Regime features — consumed by the IC meta-learner.
REGIME_FEATURES: tuple[str, ...] = (
    "regime_vol_level",
    "regime_vol_dispersion",
    "regime_trend",
    "regime_short_reversal",
    "regime_dispersion",
    "regime_sentiment",
)


def signals_for_groups(*groups: str) -> tuple[SignalDef, ...]:
    """Returns signals whose group matches any of the supplied names."""
    tokens = {g.strip().lower() for g in groups if g and g.strip()}
    if not tokens:
        return ALL_SIGNALS
    return tuple(s for s in ALL_SIGNALS if s.group in tokens)
