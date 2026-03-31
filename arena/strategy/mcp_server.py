from __future__ import annotations

from arena.strategy.catalog import get_card, list_cards, search_cards


def serve_mcp() -> None:
    """Starts a strategy-reference MCP server for agent tool usage."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "mcp package is not available. Install dependencies and retry."
        ) from exc

    mcp = FastMCP("llm-arena-strategy-reference")

    @mcp.tool()
    def list_strategy_cards() -> list[dict]:
        """Lists all strategy cards that agents can reference."""
        return list_cards()

    @mcp.tool()
    def get_strategy_card(strategy_id: str) -> dict:
        """Returns details of one strategy card by strategy_id."""
        card = get_card(strategy_id)
        if not card:
            return {"error": f"strategy_id not found: {strategy_id}"}
        return card

    @mcp.tool()
    def search_strategy_cards(keyword: str) -> list[dict]:
        """Searches strategy cards by free-text keyword."""
        return search_cards(keyword)

    @mcp.tool()
    def build_strategy_brief(strategy_ids: list[str], market_view: str) -> dict:
        """Builds a concise strategy brief from selected reference cards."""
        cards = [get_card(sid) for sid in strategy_ids]
        cards = [card for card in cards if card]
        return {
            "market_view": market_view,
            "selected_count": len(cards),
            "selected_cards": cards,
            "guidance": "Use these as references, then adapt assumptions to current data and risk limits.",
        }

    mcp.run()


if __name__ == "__main__":
    serve_mcp()
