from __future__ import annotations


def test_firestore_session_factory_uses_uri_collection(monkeypatch) -> None:
    from arena.agents.investment_chat import services

    captured: dict[str, object] = {}

    class FakeFirestoreSessionService:
        def __init__(self, *, root_collection: str | None = None) -> None:
            captured["root_collection"] = root_collection

    monkeypatch.setattr(services, "FirestoreSessionService", FakeFirestoreSessionService)

    service = services.firestore_session_factory("firestore://arena-chat-sessions")

    assert isinstance(service, FakeFirestoreSessionService)
    assert captured["root_collection"] == "arena-chat-sessions"


def test_firestore_session_factory_defaults_collection(monkeypatch) -> None:
    from arena.agents.investment_chat import services

    captured: dict[str, object] = {}

    class FakeFirestoreSessionService:
        def __init__(self, *, root_collection: str | None = None) -> None:
            captured["root_collection"] = root_collection

    monkeypatch.setattr(services, "FirestoreSessionService", FakeFirestoreSessionService)

    services.firestore_session_factory("firestore://")

    assert captured["root_collection"] == "arena-investment-chat-adk-sessions"
