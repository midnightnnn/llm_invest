from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader

from arena.agents.investment_chat import APP_NAME, build_investment_chat_agent
from arena.agents.investment_chat.context import (
    REQUEST_MODEL,
    REQUEST_PROVIDER,
    REQUEST_TENANT,
    REQUEST_USER_EMAIL,
    normalize_tenant,
)
from arena.config import Settings
from arena.providers.registry import canonical_provider, default_model_for_provider
from arena.ui.investment_chat_providers import tenant_available_provider_ids
from arena.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
CurrentUserFn = Callable[[Request], dict[str, Any] | None]

_LOADER_CACHE_MAX_ENTRIES = 64


def _encode_model_id(model_id: str) -> str:
    raw = str(model_id or "").strip().encode("utf-8")
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_model_id(encoded: str) -> str:
    token = str(encoded or "").strip()
    if not token:
        return ""
    padding = "=" * (-len(token) % 4)
    try:
        return urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")
    except Exception:
        return ""


def _chat_app_name(tenant_id: str, provider: str = "", model_id: str = "") -> str:
    tenant = re.sub(r"[^a-z0-9_-]+", "-", normalize_tenant(tenant_id)).strip("-_") or "local"
    provider_token = re.sub(r"[^a-z0-9_-]+", "-", str(provider or "").strip().lower()).strip("-_")
    model_token = _encode_model_id(model_id)
    if provider_token and model_token:
        return f"{APP_NAME}__{tenant}__{provider_token}__m_{model_token}"
    return f"{APP_NAME}__{tenant}"


def _spec_from_app_name(agent_name: str) -> tuple[str, str, str]:
    raw = str(agent_name or "").strip()
    token = raw.lower()
    prefix = f"{APP_NAME}__"
    if token.startswith(prefix):
        parts = raw[len(prefix) :].split("__")
        tenant = normalize_tenant(parts[0] if parts else "")
        provider = str(parts[1] if len(parts) >= 2 else "").strip().lower()
        model = ""
        if len(parts) >= 3 and parts[2].startswith("m_"):
            model = _decode_model_id(parts[2][2:])
        return tenant, provider, model
    return "", "", ""


def _tenant_from_app_name(agent_name: str) -> str:
    tenant, _provider, _model = _spec_from_app_name(agent_name)
    return tenant


def _adk_agents_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "agents" / "investment_chat")


class InvestmentChatAgentLoader(BaseAgentLoader):
    """ADK loader that builds the chat agent from the UI runtime repository."""

    def __init__(
        self,
        *,
        repo: Any,
        settings_for_tenant: Callable[[str], Settings],
        get_default_registry: Callable[[str], ToolRegistry],
        default_tenant: str,
    ) -> None:
        self.repo = repo
        self.settings_for_tenant = settings_for_tenant
        self.get_default_registry = get_default_registry
        self.default_tenant = normalize_tenant(default_tenant)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def _tenant_id(self, agent_name: str = "") -> str:
        encoded_tenant, _provider, _model = _spec_from_app_name(agent_name)
        requested_raw = str(os.getenv("ARENA_CHAT_TENANT_ID") or REQUEST_TENANT.get() or "").strip()
        if requested_raw:
            return normalize_tenant(requested_raw)
        if encoded_tenant:
            return encoded_tenant
        return self.default_tenant

    def _selection(self, tenant: str, agent_name: str = "") -> tuple[str, str]:
        _encoded_tenant, encoded_provider, encoded_model = _spec_from_app_name(agent_name)
        requested_provider = canonical_provider(os.getenv("ARENA_CHAT_PROVIDER") or REQUEST_PROVIDER.get()) or str(os.getenv("ARENA_CHAT_PROVIDER") or REQUEST_PROVIDER.get() or "").strip().lower()
        encoded_provider = canonical_provider(encoded_provider) or encoded_provider
        requested_model = str(os.getenv("ARENA_CHAT_MODEL") or REQUEST_MODEL.get() or "").strip()
        preferred_provider = str(requested_provider or encoded_provider or "gemini").strip().lower() or "gemini"
        allowed_providers, credential_scoped = tenant_available_provider_ids(self.repo, tenant_id=tenant)
        provider = preferred_provider
        if allowed_providers and provider not in allowed_providers:
            provider = allowed_providers[0]
        elif credential_scoped and not allowed_providers:
            return "", ""
        preferred_model = requested_model or encoded_model
        preferred_model_provider = requested_provider or encoded_provider
        model = str(preferred_model if provider == preferred_model_provider else "").strip()
        if not model:
            model = default_model_for_provider(self.settings_for_tenant(tenant), provider)
        return provider, model

    def load_agent(self, agent_name: str):
        if agent_name != APP_NAME and not _tenant_from_app_name(agent_name):
            raise ValueError(f"unknown investment chat app: {agent_name}")
        tenant = self._tenant_id(agent_name)
        provider, model_id = self._selection(tenant, agent_name)
        if not provider or not model_id:
            raise ValueError(f"no registered investment chat model provider for tenant: {tenant}")
        cache_key = f"{agent_name}:{tenant}:{provider}:{model_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        settings = self.settings_for_tenant(tenant)
        logger.info(
            "Investment chat agent selected tenant=%s provider=%s model=%s app=%s",
            tenant,
            provider,
            model_id,
            agent_name,
        )
        agent = build_investment_chat_agent(
            repo=self.repo,
            settings=settings,
            tenant_id=tenant,
            registry=None,
            provider=provider,
            model_override=model_id,
        )
        self._cache[cache_key] = agent
        if len(self._cache) > _LOADER_CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)
        return agent

    def list_agents(self) -> list[str]:
        tenant = self._tenant_id()
        provider, model_id = self._selection(tenant)
        if not provider or not model_id:
            return []
        return [_chat_app_name(tenant, provider, model_id)]

    def list_agents_detailed(self) -> list[dict[str, Any]]:
        tenant = self._tenant_id()
        provider, model_id = self._selection(tenant)
        if not provider or not model_id:
            return []
        app_name = _chat_app_name(tenant, provider, model_id)
        return [
            {
                "name": app_name,
                "root_agent_name": APP_NAME,
                "description": f"Arena 투자챗봇 ({tenant}, {provider}, {model_id})",
                "language": "python",
                "is_computer_use": False,
            }
        ]


def _copy_adk_browser_assets(*, url_prefix: str) -> Path | None:
    import google.adk.cli.fast_api as adk_fast_api

    source = Path(adk_fast_api.__file__).resolve().parent / "browser"
    if not source.exists():
        logger.warning("[yellow]ADK browser assets missing[/yellow] path=%s", source)
        return None

    try:
        adk_version = str(getattr(__import__("google.adk", fromlist=["__version__"]), "__version__", "") or "")
    except Exception:
        adk_version = ""
    digest_input = f"{url_prefix}|{adk_version}".encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:12]
    dest = Path(os.getenv("TMPDIR") or "/tmp") / "arena-adk-web-assets" / digest
    config_path = dest / "assets" / "config" / "runtime-config.json"
    if not dest.exists():
        shutil.copytree(source, dest)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            config = {}
    if config.get("backendUrl") != url_prefix:
        config["backendUrl"] = url_prefix
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _mount_adk_static(app: FastAPI, *, url_prefix: str) -> None:
    assets_dir = _copy_adk_browser_assets(url_prefix=url_prefix)
    if assets_dir is None:
        return

    redirect_url = f"{url_prefix}/dev-ui/"

    @app.get("/")
    async def redirect_root_to_dev_ui():
        return RedirectResponse(redirect_url)

    @app.get("/dev-ui")
    async def redirect_dev_ui_add_slash():
        return RedirectResponse(redirect_url)

    @app.get("/dev-ui/config")
    async def get_ui_config():
        return {"logo_text": None, "logo_image_url": None}

    app.mount(
        "/dev-ui/",
        StaticFiles(directory=str(assets_dir), html=True),
        name="investment_chat_adk_static",
    )


def _default_session_service_uri() -> str:
    explicit = str(os.getenv("ARENA_CHAT_SESSION_SERVICE_URI") or "").strip()
    if explicit:
        return explicit
    session_db_path = Path(__file__).resolve().parents[2] / "data" / "arena-investment-chat-adk-sessions.sqlite"
    session_db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{session_db_path}"


def _install_auth_gate(
    app: FastAPI,
    *,
    auth_enabled: bool,
    current_user: CurrentUserFn | None,
    default_tenant: str,
) -> None:
    @app.middleware("http")
    async def require_investment_chat_auth(request: Request, call_next):
        tenant = normalize_tenant(default_tenant)
        try:
            session_tenant = request.session.get("investment_chat_tenant_id")
        except Exception:
            session_tenant = ""
        if str(session_tenant or "").strip():
            tenant = normalize_tenant(str(session_tenant))
        query_tenant = str(request.query_params.get("tenant_id") or "").strip()
        if query_tenant and not auth_enabled:
            tenant = normalize_tenant(query_tenant)
            try:
                request.session["investment_chat_tenant_id"] = tenant
            except Exception:
                pass
        user: dict[str, Any] | None = None
        if current_user is not None:
            try:
                user = current_user(request)
            except Exception:
                user = None
        user_email = str((user or {}).get("email") or "").strip().lower()
        if not user_email and not auth_enabled:
            user_email = "local@localhost"
        provider = str(os.getenv("ARENA_CHAT_PROVIDER") or "").strip().lower()
        model = str(os.getenv("ARENA_CHAT_MODEL") or "").strip()
        if not provider:
            try:
                provider = str(request.session.get("investment_chat_provider") or "").strip().lower()
            except Exception:
                provider = ""
        if not model:
            try:
                model = str(request.session.get("investment_chat_model") or "").strip()
            except Exception:
                model = ""
        if not auth_enabled:
            query_provider = str(request.query_params.get("provider") or "").strip().lower()
            query_model = str(request.query_params.get("model") or "").strip()
            if query_provider:
                provider = query_provider
                try:
                    request.session["investment_chat_provider"] = provider
                except Exception:
                    pass
            if query_model:
                model = query_model
                try:
                    request.session["investment_chat_model"] = model
                except Exception:
                    pass
        tenant_token = REQUEST_TENANT.set(tenant)
        user_token = REQUEST_USER_EMAIL.set(user_email)
        provider_token = REQUEST_PROVIDER.set(provider)
        model_token = REQUEST_MODEL.set(model)
        try:
            if not auth_enabled:
                return await call_next(request)

            if user:
                return await call_next(request)

            accept = str(request.headers.get("accept") or "").lower()
            path = str(request.url.path or "")
            if "text/html" in accept or "/dev-ui" in path:
                return RedirectResponse("/auth/google/login", status_code=302)
            return JSONResponse({"error": "auth required"}, status_code=401)
        finally:
            REQUEST_MODEL.reset(model_token)
            REQUEST_PROVIDER.reset(provider_token)
            REQUEST_USER_EMAIL.reset(user_token)
            REQUEST_TENANT.reset(tenant_token)


def build_investment_chat_adk_app(
    *,
    repo: Any,
    settings_for_tenant: Callable[[str], Settings],
    get_default_registry: Callable[[str], ToolRegistry],
    default_tenant: str,
    url_prefix: str = "/investment-chat/adk",
    auth_enabled: bool = False,
    current_user: CurrentUserFn | None = None,
) -> FastAPI:
    loader = InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=settings_for_tenant,
        get_default_registry=get_default_registry,
        default_tenant=default_tenant,
    )
    session_service_uri = _default_session_service_uri()
    artifact_service_uri = str(os.getenv("ARENA_CHAT_ARTIFACT_SERVICE_URI") or "memory://").strip() or "memory://"
    app = get_fast_api_app(
        agents_dir=_adk_agents_dir(),
        agent_loader=loader,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=str(os.getenv("ARENA_CHAT_MEMORY_SERVICE_URI") or "").strip() or None,
        use_local_storage=False,
        allow_origins=None,
        web=False,
        url_prefix=url_prefix,
        auto_create_session=False,
    )
    _install_auth_gate(app, auth_enabled=auth_enabled, current_user=current_user, default_tenant=default_tenant)
    _mount_adk_static(app, url_prefix=url_prefix)
    return app
