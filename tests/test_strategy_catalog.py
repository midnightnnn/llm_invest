from arena.strategy.catalog import get_card, list_cards, search_cards


def test_list_cards_has_core_entries() -> None:
    """Ensures strategy catalog includes expected baseline entries."""
    cards = list_cards()
    ids = {c["strategy_id"] for c in cards}
    assert "mpt_optimize" in ids
    assert "nbeats_forecast" in ids


def test_get_card() -> None:
    """Ensures one strategy can be loaded by id."""
    card = get_card("mpt_optimize")
    assert card is not None
    assert card["category"] == "allocation"


def test_search_cards() -> None:
    """Ensures keyword search returns filtered card set."""
    cards = search_cards("regime")
    assert any(c["strategy_id"] == "macro_regime" for c in cards)
