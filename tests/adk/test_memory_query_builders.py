from __future__ import annotations

from arena.memory.query_builders import build_memory_query, build_memory_query_spec


def test_build_memory_query_screen_market_mentions_buckets_and_tickers() -> None:
    query = build_memory_query(
        "screen_market",
        {},
        [
            {"ticker": "PBR", "bucket": "value"},
            {"ticker": "MRVL", "bucket": "momentum"},
            {"ticker": "DUK", "bucket": "defensive"},
        ],
    )

    assert query == "market screening value momentum defensive PBR MRVL DUK"


def test_build_memory_query_recommend_opportunities_mentions_profiles_and_tickers() -> None:
    query = build_memory_query(
        "recommend_opportunities",
        {},
        {
            "recommendations": [
                {"ticker": "PBR", "profile": "value"},
                {"ticker": "MRVL", "profile": "aggressive"},
                {"ticker": "DUK", "profile": "defensive"},
            ],
        },
    )

    assert query == "validated opportunities value aggressive defensive PBR MRVL DUK"


def test_build_memory_query_spec_macro_snapshot_uses_regime_keys() -> None:
    spec = build_memory_query_spec(
        "macro_snapshot",
        {},
        {
            "indicators": {
                "fed_funds_rate": {"value": 5.25, "unit": "%"},
                "treasury_10y": {"value": 4.8, "unit": "%"},
                "yield_spread_10y_2y": {"value": -0.42, "unit": "pp"},
                "cpi_yoy": {"value": 3.4, "unit": "%"},
            },
            "source": "fred",
        },
    )

    assert spec is not None
    assert spec.tool_name == "macro_snapshot"
    assert spec.key_type == "regime"
    assert spec.keys == ("high_rates", "high_yields", "yield_curve_inverted", "inflation_elevated")
    assert spec.query == (
        "macro regime high_rates high_yields yield_curve_inverted inflation_elevated "
        "fed_funds_rate treasury_10y yield_spread_10y_2y cpi_yoy"
    )
    assert "key_type:regime" in spec.search_text()
    assert "regime:high_rates" in spec.search_text()


def test_build_memory_query_spec_fear_greed_uses_risk_and_volatility_regime() -> None:
    spec = build_memory_query_spec(
        "fear_greed_index",
        {},
        {
            "regime_label": "risk_off",
            "regime": "Extreme Fear",
            "volatility_index": "VIX",
            "volatility_close": 31.2,
            "fear_greed_score": 14.0,
        },
    )

    assert spec is not None
    assert spec.key_type == "regime"
    assert spec.keys == ("risk_off", "high_vol", "extreme_fear")
    assert spec.query == "market regime risk_off high_vol extreme_fear VIX"


def test_build_memory_query_spec_research_briefing_uses_theme_keys() -> None:
    spec = build_memory_query_spec(
        "get_research_briefing",
        {},
        [
            {
                "category": "global_market",
                "ticker": "NVDA",
                "headline": "AI capex plans pressure supply chain capacity",
                "summary": "Hyperscaler AI capex is still accelerating while supply chain bottlenecks persist.",
            }
        ],
    )

    assert spec is not None
    assert spec.key_type == "theme"
    assert spec.keys == ("global_market", "ai_capex", "supply_chain")
    assert spec.context_keys == ("NVDA",)
    assert spec.query == "research theme global_market ai_capex supply_chain NVDA"
    assert "theme:ai_capex" in spec.search_text()
    assert "context:NVDA" in spec.search_text()


def test_build_memory_query_spec_earnings_calendar_uses_event_class_keys() -> None:
    spec = build_memory_query_spec(
        "earnings_calendar",
        {"ticker": "AAPL"},
        {
            "ticker": "AAPL",
            "rows": [
                {"symbol": "AAPL", "event_type": "earnings", "date": "2026-05-01"},
                {"symbol": "AAPL", "event_type": "dividend", "date": "2026-05-15"},
            ],
        },
    )

    assert spec is not None
    assert spec.key_type == "event_class"
    assert spec.keys == ("earnings", "dividend")
    assert spec.context_keys == ("AAPL",)
    assert spec.query == "calendar event earnings dividend AAPL"


def test_build_memory_query_spec_index_snapshot_uses_regime_keys() -> None:
    spec = build_memory_query_spec(
        "index_snapshot",
        {},
        {
            "indices": [
                {"symbol": "VIX", "type": "index", "close": 31.0},
                {"symbol": "US10Y", "type": "bond_yield", "value": 4.8},
            ]
        },
    )

    assert spec is not None
    assert spec.key_type == "regime"
    assert spec.keys == ("risk_off", "high_vol", "market_index", "rates", "high_yields")
    assert spec.query == "market index regime risk_off high_vol market_index rates high_yields VIX US10Y"


def test_build_memory_query_spec_reddit_sentiment_uses_theme_with_ticker_context() -> None:
    spec = build_memory_query_spec(
        "fetch_reddit_sentiment",
        {"ticker": "AAPL"},
        [
            {
                "title": "AAPL AI capex debate hits wallstreetbets",
                "subreddit": "wallstreetbets",
                "selftext_snippet": "Retail sentiment is chasing AI capex beneficiaries.",
            }
        ],
    )

    assert spec is not None
    assert spec.key_type == "theme"
    assert spec.keys == ("social_sentiment", "ai_capex", "retail_sentiment")
    assert spec.context_keys == ("AAPL",)
    assert spec.query == "social sentiment AAPL ai_capex retail_sentiment"


def test_build_memory_query_spec_reddit_sentiment_uses_batch_ticker_context() -> None:
    spec = build_memory_query_spec(
        "fetch_reddit_sentiment",
        {"tickers": ["AAPL", "MSFT"]},
        {
            "tickers": ["AAPL", "MSFT"],
            "rows": [
                {
                    "ticker": "AAPL",
                    "title": "AAPL AI capex debate hits wallstreetbets",
                    "subreddit": "wallstreetbets",
                    "selftext_snippet": "Retail sentiment is chasing AI capex beneficiaries.",
                },
                {
                    "ticker": "MSFT",
                    "title": "MSFT retail sentiment stays constructive",
                    "subreddit": "stocks",
                    "selftext_snippet": "Cloud AI capex remains the key debate.",
                },
            ],
        },
    )

    assert spec is not None
    assert spec.key_type == "theme"
    assert spec.context_keys == ("AAPL", "MSFT")
    assert spec.query == "social sentiment AAPL MSFT ai_capex retail_sentiment"


def test_build_memory_query_spec_sec_filings_uses_event_class_with_entity_context() -> None:
    spec = build_memory_query_spec(
        "fetch_sec_filings",
        {"ticker": "AAPL", "filing_type": "10-K"},
        [{"form_type": "10-K", "entity": "Apple Inc.", "description": "Annual report"}],
    )

    assert spec is not None
    assert spec.key_type == "event_class"
    assert spec.keys == ("10-k",)
    assert spec.context_keys == ("AAPL", "Apple Inc.")
    assert spec.query == "filing event 10-k AAPL Apple Inc."


def test_build_memory_query_spec_sec_filings_uses_batch_ticker_context() -> None:
    spec = build_memory_query_spec(
        "fetch_sec_filings",
        {"tickers": ["AAPL", "MSFT"], "filing_type": "10-K"},
        {
            "tickers": ["AAPL", "MSFT"],
            "rows": [
                {"ticker": "AAPL", "form_type": "10-K", "entity": "Apple Inc.", "description": "Annual report"},
                {"ticker": "MSFT", "form_type": "10-K", "entity": "Microsoft Corp.", "description": "Annual report"},
            ],
        },
    )

    assert spec is not None
    assert spec.key_type == "event_class"
    assert spec.keys == ("10-k",)
    assert spec.context_keys == ("AAPL", "MSFT", "Apple Inc.", "Microsoft Corp.")
    assert spec.query == "filing event 10-k AAPL MSFT Apple Inc. Microsoft Corp."
