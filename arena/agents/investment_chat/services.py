from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlparse

from google.adk.cli.service_registry import get_service_registry
from google.adk.events.event import Event
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
from google.adk.sessions.session import Session

DEFAULT_FIRESTORE_SESSION_ROOT = "arena-investment-chat-adk-sessions"
ADK_RESERVED_SESSION_METADATA_KEY = "__session_metadata__"
FIRESTORE_SESSION_METADATA_KEY = "session_metadata"


def _firestore_session_state_for_write(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    if ADK_RESERVED_SESSION_METADATA_KEY not in state:
        return state
    safe_state = dict(state)
    metadata = safe_state.pop(ADK_RESERVED_SESSION_METADATA_KEY)
    safe_state.setdefault(FIRESTORE_SESSION_METADATA_KEY, metadata)
    return safe_state


class ArenaFirestoreSessionService(FirestoreSessionService):
    """Firestore session service that avoids ADK reserved metadata field names."""

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        return await super().create_session(
            app_name=app_name,
            user_id=user_id,
            state=_firestore_session_state_for_write(state),
            session_id=session_id,
        )

    async def append_event(self, session: Session, event: Event) -> Event:
        session.state = _firestore_session_state_for_write(session.state) or {}
        if event.actions and event.actions.state_delta:
            event.actions.state_delta = _firestore_session_state_for_write(
                event.actions.state_delta
            ) or {}
        return await super().append_event(session, event)


def _firestore_root_collection(uri: str) -> str:
    parsed = urlparse(str(uri or ""))
    raw_collection = parsed.netloc or parsed.path.strip("/")
    collection = unquote(str(raw_collection or "")).strip().strip("/")
    return collection or DEFAULT_FIRESTORE_SESSION_ROOT


def firestore_session_factory(uri: str, **kwargs):
    _ = kwargs
    return ArenaFirestoreSessionService(
        root_collection=_firestore_root_collection(uri)
    )


get_service_registry().register_session_service("firestore", firestore_session_factory)
