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

def test_sector_summary_groups() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    rows = qt.sector_summary("20d")
    assert rows
    assert "sector" in rows[0]
    assert "avg_ret" in rows[0]


def test_get_fundamentals_filters_to_target_universe() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=FakeOpenTradingClient())
    out = qt.get_fundamentals(["AAPL", "XYZ"], excd="NAS", max_items=10)
    assert out["eligible"] == ["AAPL"]
    assert out["excluded"] == ["XYZ"]
    assert out["rows"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["per"] == 31.5


def test_get_fundamentals_defaults_to_opportunity_working_set() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=FakeOpenTradingClient())
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "opportunity_working_set": [{"ticker": "MSFT", "status": "pending"}],
        }
    )

    out = qt.get_fundamentals(max_items=10)

    assert out["eligible"] == ["MSFT"]
    assert out["rows"]
    assert out["rows"][0]["ticker"] == "MSFT"


def test_get_fundamentals_normalizes_generic_us_exchange() -> None:
    client = FakeOpenTradingClient()
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=client)

    out = qt.get_fundamentals(["AAPL"], excd="US", max_items=10)

    assert out["rows"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["exchange"] == "NAS"
    assert out["rows"][0]["per"] == 31.5
    assert client.overseas_price_detail_calls == [("AAPL", "NAS")]
