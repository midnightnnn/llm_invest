from __future__ import annotations


def test_firestore_session_factory_uses_uri_collection(monkeypatch) -> None:
    from arena.agents.investment_chat import services

    captured: dict[str, object] = {}

    class FakeFirestoreSessionService:
        def __init__(self, *, root_collection: str | None = None) -> None:
            captured["root_collection"] = root_collection

    monkeypatch.setattr(services, "ArenaFirestoreSessionService", FakeFirestoreSessionService)

    service = services.firestore_session_factory("firestore://arena-chat-sessions")

    assert isinstance(service, FakeFirestoreSessionService)
    assert captured["root_collection"] == "arena-chat-sessions"


def test_firestore_session_factory_defaults_collection(monkeypatch) -> None:
    from arena.agents.investment_chat import services

    captured: dict[str, object] = {}

    class FakeFirestoreSessionService:
        def __init__(self, *, root_collection: str | None = None) -> None:
            captured["root_collection"] = root_collection

    monkeypatch.setattr(services, "ArenaFirestoreSessionService", FakeFirestoreSessionService)

    services.firestore_session_factory("firestore://")

    assert captured["root_collection"] == "arena-investment-chat-adk-sessions"


def test_firestore_session_state_renames_only_top_level_adk_reserved_metadata_key() -> None:
    from arena.agents.investment_chat import services

    state = {
        "__session_metadata__": {"displayName": "대화"},
        "nested": {"__session_metadata__": {"displayName": "하위"}},
    }

    safe_state = services._firestore_session_state_for_write(state)

    assert safe_state == {
        "session_metadata": {"displayName": "대화"},
        "nested": {"__session_metadata__": {"displayName": "하위"}},
    }
