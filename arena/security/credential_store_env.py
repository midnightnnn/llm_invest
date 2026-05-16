"""Local file/env credential store for OSS quickstart."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from arena.models import utc_now
from arena.providers.credentials import (
    build_model_secret_payload,
    normalize_provider_token,
    parse_model_secret_providers,
    runtime_credential_flags,
)


@dataclass(slots=True)
class SavedCredentialRefs:
    tenant_id: str
    updated_at: datetime
    kis_secret_name: str
    model_secret_name: str


def default_credentials_path() -> Path:
    raw = os.getenv("ARENA_LOCAL_CREDENTIALS_FILE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".llm-arena" / "credentials.json").resolve()


def load_local_secret_payload(*, secret_id: str, path: str | Path | None = None) -> dict[str, Any]:
    token = str(secret_id or "").strip()
    if not token:
        return {}
    credentials_path = Path(path).expanduser().resolve() if path is not None else default_credentials_path()
    try:
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    payload = data.get(token)
    return dict(payload) if isinstance(payload, dict) else {}


class EnvCredentialStore:
    """Stores local credentials in a private JSON file and records metadata in repo."""

    def __init__(self, *, repo: Any, path: str | Path | None = None) -> None:
        self.repo = repo
        self.path = Path(path) if path is not None else default_credentials_path()

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
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    @staticmethod
    def _secret_id(tenant_id: str, kind: str) -> str:
        tenant = str(tenant_id or "").strip().lower() or "local"
        return f"local-{tenant}-{kind}"

    @staticmethod
    def _mask(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if len(text) <= 4:
            return "*" * len(text)
        return f"{'*' * (len(text) - 4)}{text[-4:]}"

    def _latest_secret_json(self, *, secret_id: str) -> dict[str, Any]:
        return load_local_secret_payload(secret_id=secret_id, path=self.path)

    def _upsert_secret_json(self, *, secret_id: str, payload: dict[str, Any]) -> str:
        data = self._read()
        data[str(secret_id or "").strip()] = dict(payload)
        self._write(data)
        return str(secret_id or "").strip()

    def list_kis_accounts_meta(self, *, tenant_id: str) -> list[dict[str, str]]:
        payload = self._latest_secret_json(secret_id=self._secret_id(tenant_id, "kis"))
        accounts = payload.get("ACCOUNTS") or payload.get("accounts") or []
        if isinstance(accounts, dict):
            accounts = [accounts]
        if not isinstance(accounts, list):
            return []
        return [
            {
                "env": str(item.get("env") or "real"),
                "cano": str(item.get("cano") or ""),
                "prdt_cd": str(item.get("prdt_cd") or "01"),
                "app_key_masked": self._mask(str(item.get("app_key") or "")),
                "app_secret_masked": self._mask(str(item.get("app_secret") or "")),
                "paper_app_key_masked": self._mask(str(item.get("paper_app_key") or "")),
                "paper_app_secret_masked": self._mask(str(item.get("paper_app_secret") or "")),
            }
            for item in accounts
            if isinstance(item, dict) and str(item.get("cano") or "").strip()
        ]

    def model_api_key(self, *, tenant_id: str, provider: str) -> str:
        tenant = str(tenant_id or "").strip().lower() or "local"
        provider_token = normalize_provider_token(provider)
        if not provider_token:
            return ""
        model_secret_id = ""
        latest_fn = getattr(self.repo, "latest_runtime_credentials", None)
        if callable(latest_fn):
            try:
                latest = latest_fn(tenant_id=tenant) or {}
            except Exception:
                latest = {}
            model_secret_id = str(latest.get("model_secret_name") or "").strip()
        if not model_secret_id:
            model_secret_id = self._secret_id(tenant, "models")
        providers = parse_model_secret_providers(self._latest_secret_json(secret_id=model_secret_id))
        return str((providers.get(provider_token) or {}).get("api_key") or "").strip()

    def save_kis_accounts(
        self,
        *,
        tenant_id: str,
        updated_by: str,
        accounts: list[dict[str, str]],
        notes: str = "",
    ) -> SavedCredentialRefs:
        tenant = str(tenant_id or "").strip().lower() or "local"
        now = utc_now()
        kis_secret_id = self._secret_id(tenant, "kis")
        model_secret_id = self._secret_id(tenant, "models")
        previous_payload = self._latest_secret_json(secret_id=kis_secret_id)
        previous_raw = previous_payload.get("ACCOUNTS") or previous_payload.get("accounts")
        if isinstance(previous_raw, list):
            previous_accounts = [dict(item) for item in previous_raw if isinstance(item, dict)]
        elif previous_payload.get("cano"):
            previous_accounts = [previous_payload]
        else:
            previous_accounts = []
        previous_by_account = {
            (str(item.get("cano") or ""), str(item.get("prdt_cd") or "01")): item
            for item in previous_accounts
            if str(item.get("cano") or "").strip()
        }
        cleaned = []
        for account in accounts:
            item = {str(k): str(v or "").strip() for k, v in dict(account).items()}
            if item.get("account_no") and not item.get("cano"):
                digits = "".join(ch for ch in item["account_no"] if ch.isdigit())
                item["cano"] = digits[:8]
                item["prdt_cd"] = digits[8:10] or item.get("prdt_cd") or "01"
            if item.get("cano"):
                key = (str(item.get("cano") or ""), str(item.get("prdt_cd") or "01"))
                previous = previous_by_account.get(key) or {}
                cleaned.append(
                    {
                        "env": str(item.get("env") or previous.get("env") or "real").strip().lower() or "real",
                        "cano": str(item.get("cano") or ""),
                        "prdt_cd": str(item.get("prdt_cd") or "01").strip() or "01",
                        "app_key": str(item.get("app_key") or "").strip() or str(previous.get("app_key") or "").strip(),
                        "app_secret": str(item.get("app_secret") or "").strip() or str(previous.get("app_secret") or "").strip(),
                        "paper_app_key": str(item.get("paper_app_key") or "").strip() or str(previous.get("paper_app_key") or "").strip(),
                        "paper_app_secret": str(item.get("paper_app_secret") or "").strip() or str(previous.get("paper_app_secret") or "").strip(),
                    }
                )
        self._upsert_secret_json(secret_id=kis_secret_id, payload={"ACCOUNTS": cleaned, "updated_at": now.isoformat()})
        model_flags = runtime_credential_flags(parse_model_secret_providers(self._latest_secret_json(secret_id=model_secret_id)))
        upsert = getattr(self.repo, "upsert_runtime_credentials", None)
        if callable(upsert):
            upsert(
                tenant_id=tenant,
                updated_at=now,
                updated_by=updated_by,
                kis_secret_name=kis_secret_id,
                model_secret_name=model_secret_id,
                kis_account_no_masked=",".join(self._mask(str(a.get("cano") or "") + str(a.get("prdt_cd") or "")) for a in cleaned),
                kis_env=",".join(str(a.get("env") or "real") for a in cleaned),
                has_openai=model_flags["has_openai"],
                has_gemini=model_flags["has_gemini"],
                has_anthropic=model_flags["has_anthropic"],
                notes=notes,
            )
        return SavedCredentialRefs(tenant, now, kis_secret_id, model_secret_id)

    def save_model_keys(
        self,
        *,
        tenant_id: str,
        updated_by: str,
        openai_api_key: str = "",
        gemini_api_key: str = "",
        anthropic_api_key: str = "",
        providers: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        tenant = str(tenant_id or "").strip().lower() or "local"
        now = utc_now()
        model_secret_id = self._secret_id(tenant, "models")
        kis_secret_id = self._secret_id(tenant, "kis")
        previous_payload = self._latest_secret_json(secret_id=model_secret_id)
        provider_updates: dict[str, dict[str, Any]] = {
            str(provider): dict(entry)
            for provider, entry in dict(providers or {}).items()
            if str(provider or "").strip()
        }
        if str(openai_api_key or "").strip():
            provider_updates["gpt"] = {**dict(provider_updates.get("gpt") or {}), "api_key": str(openai_api_key).strip()}
        if str(gemini_api_key or "").strip():
            provider_updates["gemini"] = {**dict(provider_updates.get("gemini") or {}), "api_key": str(gemini_api_key).strip()}
        if str(anthropic_api_key or "").strip():
            provider_updates["claude"] = {**dict(provider_updates.get("claude") or {}), "api_key": str(anthropic_api_key).strip()}
        payload = build_model_secret_payload(
            previous_payload=previous_payload,
            provider_updates=provider_updates,
            updated_at=now.isoformat(),
        )
        self._upsert_secret_json(secret_id=model_secret_id, payload=payload)
        flags = runtime_credential_flags(parse_model_secret_providers(payload))
        upsert = getattr(self.repo, "upsert_runtime_credentials", None)
        if callable(upsert):
            upsert(
                tenant_id=tenant,
                updated_at=now,
                updated_by=updated_by,
                kis_secret_name=kis_secret_id,
                model_secret_name=model_secret_id,
                has_openai=flags["has_openai"],
                has_gemini=flags["has_gemini"],
                has_anthropic=flags["has_anthropic"],
                notes="local model credentials",
            )

    def remove_model_key(self, *, tenant_id: str, updated_by: str, provider: str) -> None:
        tenant = str(tenant_id or "").strip().lower() or "local"
        now = utc_now()
        model_secret_id = self._secret_id(tenant, "models")
        kis_secret_id = self._secret_id(tenant, "kis")
        previous_payload = self._latest_secret_json(secret_id=model_secret_id)
        payload = build_model_secret_payload(
            previous_payload=previous_payload,
            provider_deletes=[provider],
            updated_at=now.isoformat(),
        )
        self._upsert_secret_json(secret_id=model_secret_id, payload=payload)
        flags = runtime_credential_flags(parse_model_secret_providers(payload))
        upsert = getattr(self.repo, "upsert_runtime_credentials", None)
        if callable(upsert):
            upsert(
                tenant_id=tenant,
                updated_at=now,
                updated_by=updated_by,
                kis_secret_name=kis_secret_id,
                model_secret_name=model_secret_id,
                has_openai=flags["has_openai"],
                has_gemini=flags["has_gemini"],
                has_anthropic=flags["has_anthropic"],
                notes="local model credentials",
            )
