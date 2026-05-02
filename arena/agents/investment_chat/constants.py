from __future__ import annotations

APP_NAME = "investment_chat"
AGENT_ID = "investment_chat"

CHAT_ANALYSIS_TOOL_IDS = frozenset(
    {
        "search_past_experiences",
        "search_peer_lessons",
        "get_research_briefing",
        "portfolio_diagnosis",
        "trade_performance",
        "recommend_opportunities",
        "screen_market",
        "optimize_portfolio",
        "forecast_returns",
        "technical_signals",
        "sector_summary",
        "get_fundamentals",
        "index_snapshot",
        "fear_greed_index",
        "earnings_calendar",
        "fetch_reddit_sentiment",
        "fetch_sec_filings",
        "macro_snapshot",
    }
)

WRITE_TOOL_MARKERS = frozenset(
    {
        "execute",
        "submit",
        "place_order",
        "broker",
        "sync_account",
        "write_",
        "delete_",
        "upsert_",
    }
)
