from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StrategyCard:
    """Represents a strategy reference that agents can cite or adapt."""

    strategy_id: str
    name: str
    category: str
    hypothesis: str
    core_inputs: list[str]
    implementation_hint: str
    main_risks: list[str]


CATALOG: tuple[StrategyCard, ...] = (
    StrategyCard(
        strategy_id="mpt_optimize",
        name="Mean-Variance MPT (Max-Sharpe)",
        category="allocation",
        hypothesis="Risk-adjusted allocation can improve portfolio efficiency.",
        core_inputs=["expected_return", "covariance", "risk_free_rate"],
        implementation_hint="Sanitize covariance to PSD and solve long-only max-Sharpe with constraints.",
        main_risks=["unstable expected returns", "overfitting on short windows"],
    ),
    StrategyCard(
        strategy_id="min_vol_allocation",
        name="Minimum Volatility",
        category="allocation",
        hypothesis="Minimizing variance yields robust risk control when expected returns are noisy.",
        core_inputs=["covariance", "constraints (long-only)"],
        implementation_hint="Use PSD-sanitized covariance; add max-weight cap if concentration occurs.",
        main_risks=["concentration in low-vol names", "underperformance in strong risk-on regimes"],
    ),
    StrategyCard(
        strategy_id="hrp_allocation",
        name="Hierarchical Risk Parity (HRP)",
        category="allocation",
        hypothesis="Hierarchical clustering + risk parity can improve diversification without relying on unstable mu.",
        core_inputs=["correlation", "covariance", "returns window"],
        implementation_hint="Convert correlation to distance, run linkage, then recursive bisection with cluster variances.",
        main_risks=["unstable clusters on short windows", "correlation regime shifts"],
    ),
    StrategyCard(
        strategy_id="blend_sharpe_hrp",
        name="Blend: Max-Sharpe + HRP",
        category="allocation",
        hypothesis="Blending a return-seeking optimizer with a risk-budgeting allocator can improve robustness.",
        core_inputs=["expected_return", "covariance", "blend_ratio"],
        implementation_hint="Compute both allocations on same window, blend weights, then re-normalize.",
        main_risks=["objective conflict", "turnover and transaction costs"],
    ),
    StrategyCard(
        strategy_id="forecast_max_sharpe",
        name="Forecast-Aware Max-Sharpe",
        category="allocation",
        hypothesis="Blending external forecasts for mu with historical estimates can improve efficiency vs sample mean.",
        core_inputs=["forecast mu (annual)", "historical returns", "covariance", "mu_confidence"],
        implementation_hint="Blend mu = (1-c)*mu_hist + c*mu_forecast, then solve long-only max-Sharpe.",
        main_risks=["forecast model drift", "overconfidence in predictions"],
    ),
    StrategyCard(
        strategy_id="nbeats_forecast",
        name="Stacked Return Forecast",
        category="forecasting",
        hypothesis="Stacked ensemble forecasts provide more robust expected returns than a single model.",
        core_inputs=["base model forecasts", "daily prices", "dividend series", "FX series"],
        implementation_hint="Use stacked/meta forecast outputs for annualized expected return inputs.",
        main_risks=["regime shifts", "meta-model drift"],
    ),
    StrategyCard(
        strategy_id="momentum_126",
        name="126-Day Momentum",
        category="signal",
        hypothesis="Assets with stronger medium-term trends continue to outperform.",
        core_inputs=["ret_20d", "ret_60d", "ret_126d"],
        implementation_hint="Use volatility scaling before ranking candidates.",
        main_risks=["sharp reversals", "crowded trades"],
    ),
    StrategyCard(
        strategy_id="mean_reversion_rsi",
        name="RSI Mean Reversion",
        category="signal",
        hypothesis="Short-term overreaction tends to revert in liquid names.",
        core_inputs=["RSI", "ATR", "volume"],
        implementation_hint="Trigger only under confirmed range-bound regimes.",
        main_risks=["catching falling knives", "false reversals"],
    ),
    StrategyCard(
        strategy_id="quality_value",
        name="Quality Value Composite",
        category="fundamental",
        hypothesis="High-quality, undervalued names outperform long horizon.",
        core_inputs=["ROE", "FCF margin", "EV/EBIT"],
        implementation_hint="Normalize by sector before computing composite score.",
        main_risks=["value traps", "slow catalyst realization"],
    ),
    StrategyCard(
        strategy_id="vol_targeting",
        name="Volatility Targeting",
        category="risk",
        hypothesis="Keeping risk budget stable improves compounding consistency.",
        core_inputs=["realized_vol_20d", "target_vol"],
        implementation_hint="Scale gross exposure with floor and cap constraints.",
        main_risks=["vol lag", "procyclical deleveraging"],
    ),
    StrategyCard(
        strategy_id="pairs_spread",
        name="Pairs Spread Reversion",
        category="stat-arb",
        hypothesis="Co-integrated pairs mean-revert around stable spreads.",
        core_inputs=["pair beta", "z-score", "spread half-life"],
        implementation_hint="Use rolling hedge ratio and stop-loss on z-score drift.",
        main_risks=["structural breaks", "execution slippage"],
    ),
    StrategyCard(
        strategy_id="macro_regime",
        name="Macro Regime Rotation",
        category="macro",
        hypothesis="Regime-aware allocation outperforms static mix.",
        core_inputs=["yield curve", "VIX", "USD strength", "inflation"],
        implementation_hint="Blend regime probabilities with base allocations.",
        main_risks=["late regime detection", "policy shocks"],
    ),
)


def list_cards() -> list[dict]:
    """Returns all strategy cards as dictionaries for tool responses."""
    return [asdict(card) for card in CATALOG]


def get_card(strategy_id: str) -> dict | None:
    """Returns one strategy card by identifier."""
    for card in CATALOG:
        if card.strategy_id == strategy_id:
            return asdict(card)
    return None


def search_cards(keyword: str) -> list[dict]:
    """Returns cards matching a simple keyword search."""
    key = keyword.strip().lower()
    if not key:
        return list_cards()
    results = []
    for card in CATALOG:
        haystack = " ".join(
            [
                card.strategy_id,
                card.name,
                card.category,
                card.hypothesis,
                card.implementation_hint,
                " ".join(card.core_inputs),
                " ".join(card.main_risks),
            ]
        ).lower()
        if key in haystack:
            results.append(asdict(card))
    return results
