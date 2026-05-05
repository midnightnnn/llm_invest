from __future__ import annotations

from datetime import datetime, timezone
import math
from types import SimpleNamespace
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    _literal_args,
    _stable_quant_tool_now,
    FakeRepo,
    FakeOpenTradingClient,
    _settings,
)

def test_quant_tool_choice_parameters_are_schema_literals() -> None:
    hints = get_type_hints(QuantTools.screen_market)
    assert _literal_args(hints["bucket"]) >= {
        "balanced",
        "momentum",
        "pullback",
        "recovery",
        "defensive",
        "value",
    }
    assert _literal_args(hints["order"]) == {"asc", "desc"}

    optimize_hints = get_type_hints(QuantTools.optimize_portfolio)
    assert _literal_args(optimize_hints["strategy"]) == {"sharpe", "risk_parity", "forecast"}

    forecast_hints = get_type_hints(QuantTools.forecast_returns)
    assert _literal_args(forecast_hints["forecast_mode"]) >= {"all", "stacked", "base", "balanced"}

    index_hints = get_type_hints(QuantTools.index_snapshot)
    assert {"SPX", "COMP", "DJI", "US10Y", "GOLD", "KOSPI"} <= _literal_args(index_hints["indices"])
