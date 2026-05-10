from __future__ import annotations

from urllib.parse import unquote, urlparse

from google.adk.cli.service_registry import get_service_registry
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService

DEFAULT_FIRESTORE_SESSION_ROOT = "arena-investment-chat-adk-sessions"


def _firestore_root_collection(uri: str) -> str:
    parsed = urlparse(str(uri or ""))
    raw_collection = parsed.netloc or parsed.path.strip("/")
    collection = unquote(str(raw_collection or "")).strip().strip("/")
    return collection or DEFAULT_FIRESTORE_SESSION_ROOT


def firestore_session_factory(uri: str, **kwargs):
    _ = kwargs
    return FirestoreSessionService(root_collection=_firestore_root_collection(uri))


get_service_registry().register_session_service("firestore", firestore_session_factory)
