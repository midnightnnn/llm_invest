from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """Represents one cached OAuth token."""

    token: str
    expires_at: datetime


def _doc_id(*, base_url: str, app_key: str) -> str:
    """Builds a stable Firestore doc id without storing the raw app key."""
    raw = f"{base_url}|{app_key}".encode("utf-8")
    return "kis_token_" + hashlib.sha256(raw).hexdigest()[:24]


class FirestoreTokenCache:
    """Firestore-backed token cache shared across Cloud Run executions."""

    def __init__(self, *, project: str, collection: str = "api_tokens"):
        self.project = project
        self.collection = collection

    def _client(self):
        from google.cloud import firestore

        return firestore.Client(project=self.project)

    def get(self, *, base_url: str, app_key: str) -> TokenRecord | None:
        """Returns cached token when present and unexpired."""
        doc = self._client().collection(self.collection).document(_doc_id(base_url=base_url, app_key=app_key))
        try:
            snap = doc.get()
        except Exception as exc:
            logger.warning("[yellow]Token cache read failed[/yellow] err=%s", str(exc))
            return None

        if not getattr(snap, "exists", False):
            return None
        data: dict[str, Any] = snap.to_dict() or {}
        token = str(data.get("token", "") or "").strip()
        expires_at = data.get("expires_at")
        if not token or not isinstance(expires_at, datetime):
            return None
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            return None
        return TokenRecord(token=token, expires_at=expires_at)

    def set(self, *, base_url: str, app_key: str, record: TokenRecord) -> None:
        """Stores token record."""
        doc = self._client().collection(self.collection).document(_doc_id(base_url=base_url, app_key=app_key))
        payload = {
            "token": record.token,
            "expires_at": record.expires_at,
            "updated_at": datetime.now(timezone.utc),
        }
        try:
            doc.set(payload)
        except Exception as exc:
            logger.warning("[yellow]Token cache write failed[/yellow] err=%s", str(exc))
