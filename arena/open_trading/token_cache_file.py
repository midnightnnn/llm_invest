"""File-backed KIS token cache for local mode."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any

from arena.open_trading.token_cache import TokenRecord, _doc_id


def default_token_cache_path() -> Path:
    raw = os.getenv("KIS_TOKEN_CACHE_FILE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".llm-arena" / "tokens.json").resolve()


class FileTokenCache:
    """Small atomic JSON token cache for single-machine local runs."""

    def __init__(self, *, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_token_cache_path()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def get(self, *, base_url: str, app_key: str) -> TokenRecord | None:
        key = _doc_id(base_url=base_url, app_key=app_key)
        row = self._read().get(key)
        if not isinstance(row, dict):
            return None
        token = str(row.get("token") or "").strip()
        expires_raw = str(row.get("expires_at") or "").strip()
        if not token or not expires_raw:
            return None
        try:
            expires_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            return None
        return TokenRecord(token=token, expires_at=expires_at)

    def set(self, *, base_url: str, app_key: str, record: TokenRecord) -> None:
        data = self._read()
        data[_doc_id(base_url=base_url, app_key=app_key)] = {
            "token": record.token,
            "expires_at": record.expires_at.astimezone(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(data)
