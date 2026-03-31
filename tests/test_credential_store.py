from __future__ import annotations

import json

import pytest
from google.api_core.exceptions import NotFound

from arena.security.credential_store import CredentialStore


class _DummyRepo:
    def __init__(self, latest: dict | None = None):
        self._latest = latest or {}
        self.upserts: list[dict] = []

    def upsert_runtime_credentials(self, **kwargs):
        self.upserts.append(dict(kwargs))

    def latest_runtime_credentials(self, *, tenant_id: str):
        _ = tenant_id
        return dict(self._latest)


def test_latest_secret_json_returns_empty_on_not_found(monkeypatch) -> None:
    class _FakeClient:
        def access_secret_version(self, request):
            _ = request
            raise NotFound("missing")

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(project="proj", repo=_DummyRepo())

    assert store._latest_secret_json(secret_id="tenant-kis") == {}


def test_latest_secret_json_raises_on_read_error(monkeypatch) -> None:
    class _FakeClient:
        def access_secret_version(self, request):
            _ = request
            raise RuntimeError("unavailable")

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(project="proj", repo=_DummyRepo())

    with pytest.raises(RuntimeError, match="failed to read secret payload"):
        store._latest_secret_json(secret_id="tenant-kis")


def test_latest_secret_json_raises_on_invalid_payload(monkeypatch) -> None:
    class _FakeClient:
        def access_secret_version(self, request):
            _ = request

            class _Resp:
                class payload:  # noqa: N801
                    data = json.dumps(["not", "object"]).encode("utf-8")

            return _Resp()

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(project="proj", repo=_DummyRepo())

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        store._latest_secret_json(secret_id="tenant-kis")


def test_list_kis_accounts_meta_skips_secret_lookup_without_runtime_secret_ref(monkeypatch) -> None:
    class _FakeClient:
        def access_secret_version(self, request):
            raise AssertionError(f"unexpected Secret Manager call: {request}")

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(project="proj", repo=_DummyRepo())

    assert store.list_kis_accounts_meta(tenant_id="local") == []


def test_list_kis_accounts_meta_uses_runtime_secret_name(monkeypatch) -> None:
    seen: list[str] = []

    class _FakeClient:
        def access_secret_version(self, request):
            seen.append(str(request.get("name") or ""))

            class _Resp:
                class payload:  # noqa: N801
                    data = json.dumps(
                        {
                            "ACCOUNTS": [
                                {"env": "real", "cano": "12345678", "prdt_cd": "01"},
                            ]
                        }
                    ).encode("utf-8")

            return _Resp()

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(
        project="proj",
        repo=_DummyRepo(latest={"kis_secret_name": "custom-kis-secret"}),
    )

    rows = store.list_kis_accounts_meta(tenant_id="local")

    assert rows == [
        {
            "env": "real",
            "cano": "12345678",
            "prdt_cd": "01",
            "app_key_masked": "",
            "app_secret_masked": "",
            "paper_app_key_masked": "",
            "paper_app_secret_masked": "",
        }
    ]
    assert seen == ["projects/proj/secrets/custom-kis-secret/versions/latest"]


def test_list_kis_accounts_meta_masks_saved_keys(monkeypatch) -> None:
    class _FakeClient:
        def access_secret_version(self, request):
            _ = request

            class _Resp:
                class payload:  # noqa: N801
                    data = json.dumps(
                        {
                            "ACCOUNTS": [
                                {
                                    "env": "real",
                                    "cano": "12345678",
                                    "prdt_cd": "01",
                                    "app_key": "abcd1234wxyz",
                                    "app_secret": "secret-value-9876",
                                    "paper_app_key": "paperkey123456",
                                    "paper_app_secret": "papersecret987654",
                                }
                            ]
                        }
                    ).encode("utf-8")

            return _Resp()

    monkeypatch.setattr(
        "arena.security.credential_store.secretmanager.SecretManagerServiceClient",
        lambda: _FakeClient(),
    )
    store = CredentialStore(
        project="proj",
        repo=_DummyRepo(latest={"kis_secret_name": "custom-kis-secret"}),
    )

    rows = store.list_kis_accounts_meta(tenant_id="local")

    assert rows[0]["app_key_masked"] == "abcd****wxyz"
    assert rows[0]["app_secret_masked"].startswith("secr")
    assert rows[0]["app_secret_masked"].endswith("9876")
    assert rows[0]["paper_app_key_masked"].startswith("pape")
    assert rows[0]["paper_app_secret_masked"].endswith("7654")


def test_save_model_keys_writes_provider_payload_and_preserves_existing_flags(monkeypatch) -> None:
    repo = _DummyRepo()
    store = CredentialStore(project="proj", repo=repo)
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        store,
        "_latest_secret_json",
        lambda *, secret_id: (
            {
                "providers": {
                    "gpt": {"api_key": "prev-openai"},
                }
            }
            if secret_id.endswith("models")
            else {}
        ),
    )
    monkeypatch.setattr(
        store,
        "_upsert_secret_json",
        lambda *, secret_id, payload: writes.append((secret_id, dict(payload))) or secret_id,
    )

    store.save_model_keys(
        tenant_id="local",
        updated_by="tester",
        providers={"gemini": {"api_key": "tenant-gemini"}},
    )

    assert writes
    _, payload = writes[-1]
    assert payload["providers"]["gpt"]["api_key"] == "prev-openai"
    assert payload["providers"]["gemini"]["api_key"] == "tenant-gemini"
    assert payload["openai_api_key"] == "prev-openai"
    assert payload["gemini_api_key"] == "tenant-gemini"
    assert payload["anthropic_api_key"] == ""
    assert repo.upserts[-1]["has_openai"] is True
    assert repo.upserts[-1]["has_gemini"] is True
    assert repo.upserts[-1]["has_anthropic"] is False


def test_save_model_keys_preserves_kis_metadata(monkeypatch) -> None:
    repo = _DummyRepo(latest={
        "kis_account_no_masked": "****5678",
        "kis_env": "real",
        "notes": "prev note",
    })
    store = CredentialStore(project="proj", repo=repo)
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        store,
        "_latest_secret_json",
        lambda *, secret_id: (
            {"providers": {"gpt": {"api_key": "key123"}}}
            if secret_id.endswith("models")
            else {}
        ),
    )
    monkeypatch.setattr(
        store,
        "_upsert_secret_json",
        lambda *, secret_id, payload: writes.append((secret_id, dict(payload))) or secret_id,
    )

    store.save_model_keys(
        tenant_id="freebong3072",
        updated_by="admin",
        openai_api_key="key123",
    )

    row = repo.upserts[-1]
    assert row["kis_account_no_masked"] == "****5678"
    assert row["kis_env"] == "real"
    assert row["notes"] == "prev note"
    assert row["has_openai"] is True
