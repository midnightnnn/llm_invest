from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import secretmanager

from arena.data.bq import BigQueryRepository
from arena.models import utc_now
from arena.providers.credentials import build_model_secret_payload, parse_model_secret_providers, runtime_credential_flags


@dataclass(slots=True)
class SavedCredentialRefs:
    """References returned after a successful credential save."""

    tenant_id: str
    updated_at: datetime
    kis_secret_name: str
    model_secret_name: str


class CredentialStore:
    """Stores broker/model credentials in Secret Manager and metadata in BigQuery."""

    def __init__(self, *, project: str, repo: BigQueryRepository, secret_prefix: str = "llm-arena"):
        self.project = str(project or "").strip()
        self.repo = repo
        self.secret_prefix = str(secret_prefix or "llm-arena").strip().lower()
        if not self.project:
            raise ValueError("project is required")
        self.client = secretmanager.SecretManagerServiceClient()

    def _safe_token(self, value: str) -> str:
        token = re.sub(r"[^a-zA-Z0-9-]", "-", str(value or "").strip().lower())
        token = re.sub(r"-+", "-", token).strip("-")
        return token or "default"

    def _secret_id(self, tenant_id: str, kind: str) -> str:
        return f"{self._safe_token(self.secret_prefix)}-{self._safe_token(tenant_id)}-{self._safe_token(kind)}"

    def _upsert_secret_json(self, *, secret_id: str, payload: dict[str, Any]) -> str:
        parent = f"projects/{self.project}"
        name = f"{parent}/secrets/{secret_id}"
        try:
            self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        except AlreadyExists:
            pass

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.client.add_secret_version(
            request={
                "parent": name,
                "payload": {"data": body},
            }
        )
        return secret_id

    def _latest_secret_json(self, *, secret_id: str) -> dict[str, Any]:
        name = f"projects/{self.project}/secrets/{secret_id}/versions/latest"
        try:
            response = self.client.access_secret_version(request={"name": name})
        except NotFound:
            return {}
        except Exception as exc:
            raise RuntimeError(f"failed to read secret payload: {name}") from exc

        payload_text = response.payload.data.decode("utf-8")
        try:
            data = json.loads(payload_text)
        except Exception as exc:
            raise RuntimeError(f"failed to parse secret payload as JSON: {name}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"secret payload must be a JSON object: {name}")
        return data

    def _split_account_no(self, account_no: str, default_prdt_cd: str = "01") -> tuple[str, str]:
        digits = re.sub(r"\D", "", str(account_no or ""))
        if len(digits) >= 10:
            return digits[:8], digits[8:10]
        if len(digits) == 8:
            return digits, str(default_prdt_cd or "01")
        return "", str(default_prdt_cd or "01")

    def _mask_account_no(self, account_no: str) -> str:
        digits = re.sub(r"\D", "", str(account_no or ""))
        if len(digits) < 4:
            return ""
        return f"{'*' * max(0, len(digits) - 4)}{digits[-4:]}"

    def _mask_secret_value(self, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        if len(token) <= 8:
            return "*" * len(token)
        head = token[:4]
        tail = token[-4:]
        return f"{head}{'*' * max(4, len(token) - 8)}{tail}"

    def list_kis_accounts_meta(self, *, tenant_id: str) -> list[dict[str, str]]:
        """Return metadata-only list of KIS accounts from Secret Manager (no secrets)."""
        kis_secret_id = ""
        latest_runtime_credentials = getattr(self.repo, "latest_runtime_credentials", None)
        if callable(latest_runtime_credentials):
            try:
                latest = latest_runtime_credentials(tenant_id=tenant_id) or {}
            except Exception:
                latest = {}
            kis_secret_id = str(latest.get("kis_secret_name") or "").strip()
            if not kis_secret_id:
                return []
        if not kis_secret_id:
            kis_secret_id = self._secret_id(tenant_id, "kis")
        prev = self._latest_secret_json(secret_id=kis_secret_id)
        raw = prev.get("ACCOUNTS") or prev.get("accounts")
        if isinstance(raw, list):
            accounts = raw
        elif prev.get("cano"):
            accounts = [prev]
        else:
            return []
        return [
            {
                "env": str(a.get("env") or "real"),
                "cano": str(a.get("cano") or ""),
                "prdt_cd": str(a.get("prdt_cd") or "01"),
                "app_key_masked": self._mask_secret_value(str(a.get("app_key") or "")),
                "app_secret_masked": self._mask_secret_value(str(a.get("app_secret") or "")),
                "paper_app_key_masked": self._mask_secret_value(str(a.get("paper_app_key") or "")),
                "paper_app_secret_masked": self._mask_secret_value(str(a.get("paper_app_secret") or "")),
            }
            for a in accounts
            if isinstance(a, dict) and str(a.get("cano") or "").strip()
        ]

    def save_kis_accounts(
        self,
        *,
        tenant_id: str,
        updated_by: str,
        accounts: list[dict[str, str]],
        notes: str = "",
    ) -> SavedCredentialRefs:
        """Save multiple KIS accounts into a single Secret Manager secret."""
        tenant = self._safe_token(tenant_id)
        now = utc_now()

        kis_secret_id = self._secret_id(tenant, "kis")
        model_secret_id = self._secret_id(tenant, "models")
        prev_kis = self._latest_secret_json(secret_id=kis_secret_id)

        # Parse previous ACCOUNTS array
        raw_prev = prev_kis.get("ACCOUNTS") or prev_kis.get("accounts")
        if isinstance(raw_prev, list):
            prev_accounts = [a for a in raw_prev if isinstance(a, dict)]
        elif prev_kis.get("cano"):
            prev_accounts = [prev_kis]
        else:
            prev_accounts = []
        prev_map = {
            (str(a.get("cano") or ""), str(a.get("prdt_cd") or "01")): a
            for a in prev_accounts
        }

        merged: list[dict[str, str]] = []
        for acct in accounts:
            cano, prdt_cd = self._split_account_no(
                str(acct.get("account_no") or acct.get("cano") or ""),
                str(acct.get("prdt_cd") or "01"),
            )
            if not cano:
                continue
            prev_acct = prev_map.get((cano, prdt_cd)) or {}
            merged.append({
                "env": str(acct.get("env") or "real").strip().lower() or "real",
                "cano": cano,
                "prdt_cd": prdt_cd,
                "app_key": str(acct.get("app_key") or "").strip() or str(prev_acct.get("app_key") or "").strip(),
                "app_secret": str(acct.get("app_secret") or "").strip() or str(prev_acct.get("app_secret") or "").strip(),
                "paper_app_key": str(acct.get("paper_app_key") or "").strip() or str(prev_acct.get("paper_app_key") or "").strip(),
                "paper_app_secret": str(acct.get("paper_app_secret") or "").strip() or str(prev_acct.get("paper_app_secret") or "").strip(),
            })

        kis_payload: dict[str, Any] = {
            "ACCOUNTS": merged,
            "updated_at": now.isoformat(),
        }
        self._upsert_secret_json(secret_id=kis_secret_id, payload=kis_payload)

        prev_model_providers = parse_model_secret_providers(self._latest_secret_json(secret_id=model_secret_id))
        model_flags = runtime_credential_flags(prev_model_providers)

        masked_parts = [self._mask_account_no(a["cano"] + a["prdt_cd"]) for a in merged]
        masked_csv = ",".join(masked_parts) if masked_parts else ""

        self.repo.upsert_runtime_credentials(
            tenant_id=tenant,
            updated_at=now,
            updated_by=updated_by,
            kis_secret_name=kis_secret_id,
            model_secret_name=model_secret_id,
            kis_account_no_masked=masked_csv,
            kis_env=",".join(a["env"] for a in merged) if merged else "",
            has_openai=model_flags["has_openai"],
            has_gemini=model_flags["has_gemini"],
            has_anthropic=model_flags["has_anthropic"],
            notes=notes,
        )

        return SavedCredentialRefs(
            tenant_id=tenant,
            updated_at=now,
            kis_secret_name=kis_secret_id,
            model_secret_name=model_secret_id,
        )

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
        """Save LLM model API keys to Secret Manager without touching KIS secret."""
        tenant = self._safe_token(tenant_id)
        now = utc_now()
        model_secret_id = self._secret_id(tenant, "models")
        prev_model = self._latest_secret_json(secret_id=model_secret_id)

        provider_updates = {
            str(provider or "").strip().lower(): dict(entry)
            for provider, entry in dict(providers or {}).items()
            if str(provider or "").strip()
        }
        if str(openai_api_key or "").strip():
            provider_updates["gpt"] = {"api_key": str(openai_api_key).strip(), **dict(provider_updates.get("gpt") or {})}
            provider_updates["gpt"]["api_key"] = str(openai_api_key).strip()
        if str(gemini_api_key or "").strip():
            provider_updates["gemini"] = {"api_key": str(gemini_api_key).strip(), **dict(provider_updates.get("gemini") or {})}
            provider_updates["gemini"]["api_key"] = str(gemini_api_key).strip()
        if str(anthropic_api_key or "").strip():
            provider_updates["claude"] = {"api_key": str(anthropic_api_key).strip(), **dict(provider_updates.get("claude") or {})}
            provider_updates["claude"]["api_key"] = str(anthropic_api_key).strip()

        model_payload = build_model_secret_payload(
            previous_payload=prev_model,
            provider_updates=provider_updates,
            updated_at=now.isoformat(),
        )
        self._upsert_secret_json(secret_id=model_secret_id, payload=model_payload)
        model_flags = runtime_credential_flags(parse_model_secret_providers(model_payload))

        kis_secret_id = self._secret_id(tenant, "kis")

        # Preserve existing KIS metadata from previous runtime_credentials row
        prev_cred: dict[str, Any] = {}
        latest_fn = getattr(self.repo, "latest_runtime_credentials", None)
        if callable(latest_fn):
            try:
                prev_cred = latest_fn(tenant_id=tenant) or {}
            except Exception:
                prev_cred = {}

        self.repo.upsert_runtime_credentials(
            tenant_id=tenant,
            updated_at=now,
            updated_by=updated_by,
            kis_secret_name=kis_secret_id,
            model_secret_name=model_secret_id,
            kis_account_no_masked=str(prev_cred.get("kis_account_no_masked") or ""),
            kis_env=str(prev_cred.get("kis_env") or ""),
            has_openai=model_flags["has_openai"],
            has_gemini=model_flags["has_gemini"],
            has_anthropic=model_flags["has_anthropic"],
            notes=str(prev_cred.get("notes") or ""),
        )

    def save_runtime_credentials(
        self,
        *,
        tenant_id: str,
        updated_by: str,
        kis_env: str,
        kis_account_no: str,
        kis_app_key: str,
        kis_app_secret: str,
        kis_paper_app_key: str,
        kis_paper_app_secret: str,
        openai_api_key: str,
        gemini_api_key: str,
        anthropic_api_key: str,
        providers: dict[str, dict[str, Any]] | None = None,
        notes: str = "",
    ) -> SavedCredentialRefs:
        """Writes credential payloads and metadata, then returns new references."""
        tenant = self._safe_token(tenant_id)
        now = utc_now()

        kis_secret_id = self._secret_id(tenant, "kis")
        model_secret_id = self._secret_id(tenant, "models")
        prev_kis = self._latest_secret_json(secret_id=kis_secret_id)
        prev_model = self._latest_secret_json(secret_id=model_secret_id)

        cano, prdt_cd = self._split_account_no(kis_account_no)
        if not cano:
            cano = str(prev_kis.get("cano") or "").strip()
            prdt_cd = str(prev_kis.get("prdt_cd") or prdt_cd).strip() or "01"

        kis_payload = {
            "app_key": str(kis_app_key or "").strip() or str(prev_kis.get("app_key") or "").strip(),
            "app_secret": str(kis_app_secret or "").strip() or str(prev_kis.get("app_secret") or "").strip(),
            "paper_app_key": str(kis_paper_app_key or "").strip() or str(prev_kis.get("paper_app_key") or "").strip(),
            "paper_app_secret": str(kis_paper_app_secret or "").strip() or str(prev_kis.get("paper_app_secret") or "").strip(),
            "cano": cano,
            "prdt_cd": prdt_cd,
            "updated_at": now.isoformat(),
        }
        provider_updates = {
            str(provider or "").strip().lower(): dict(entry)
            for provider, entry in dict(providers or {}).items()
            if str(provider or "").strip()
        }
        if str(openai_api_key or "").strip():
            provider_updates["gpt"] = {"api_key": str(openai_api_key).strip(), **dict(provider_updates.get("gpt") or {})}
            provider_updates["gpt"]["api_key"] = str(openai_api_key).strip()
        if str(gemini_api_key or "").strip():
            provider_updates["gemini"] = {"api_key": str(gemini_api_key).strip(), **dict(provider_updates.get("gemini") or {})}
            provider_updates["gemini"]["api_key"] = str(gemini_api_key).strip()
        if str(anthropic_api_key or "").strip():
            provider_updates["claude"] = {"api_key": str(anthropic_api_key).strip(), **dict(provider_updates.get("claude") or {})}
            provider_updates["claude"]["api_key"] = str(anthropic_api_key).strip()

        model_payload = build_model_secret_payload(
            previous_payload=prev_model,
            provider_updates=provider_updates,
            updated_at=now.isoformat(),
        )

        # Keep one versioned secret per tenant/type and rely on Secret Manager history.
        self._upsert_secret_json(secret_id=kis_secret_id, payload=kis_payload)
        self._upsert_secret_json(secret_id=model_secret_id, payload=model_payload)
        model_flags = runtime_credential_flags(parse_model_secret_providers(model_payload))

        self.repo.upsert_runtime_credentials(
            tenant_id=tenant,
            updated_at=now,
            updated_by=updated_by,
            kis_secret_name=kis_secret_id,
            model_secret_name=model_secret_id,
            kis_account_no_masked=self._mask_account_no(kis_account_no),
            kis_env=(kis_env or "").strip().lower(),
            has_openai=model_flags["has_openai"],
            has_gemini=model_flags["has_gemini"],
            has_anthropic=model_flags["has_anthropic"],
            notes=notes,
        )

        return SavedCredentialRefs(
            tenant_id=tenant,
            updated_at=now,
            kis_secret_name=kis_secret_id,
            model_secret_name=model_secret_id,
        )
