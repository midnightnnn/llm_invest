from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

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
from arena.agents.investment_chat.config_tools import load_chat_agent_config
from arena.agents.investment_chat.selection import (
    normalize_chat_model_selection,
    normalize_stored_advisor_model_selection,
    tenant_default_chat_selection,
)
from arena.config import Settings
from arena.providers.registry import (
    canonical_provider,
    default_model_for_provider,
    provider_api_key_from_settings,
    provider_base_url_from_settings,
    provider_has_credentials,
)
from arena.ui.investment_chat_providers import tenant_available_provider_ids
from arena.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
CurrentUserFn = Callable[[Request], dict[str, Any] | None]

_LOADER_CACHE_MAX_ENTRIES = 64


class _SuppressAdkExperimentalWarnings:
    _ENV_KEY = "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS"

    def __enter__(self) -> None:
        self._previous = os.environ.get(self._ENV_KEY)
        os.environ[self._ENV_KEY] = "true"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._previous is None:
            os.environ.pop(self._ENV_KEY, None)
        else:
            os.environ[self._ENV_KEY] = self._previous


def _secretish(name: str) -> bool:
    token = str(name or "").strip().lower()
    return any(marker in token for marker in ("key", "secret", "token", "password"))


def _secret_digest(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _fingerprint_value(name: str, value: Any) -> Any:
    if _secretish(name):
        return _secret_digest(value)
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(str(key), item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_fingerprint_value(name, item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _repo_config_value(repo: Any, tenant_id: str, key: str) -> str:
    loader = getattr(repo, "get_config", None)
    if not callable(loader):
        return ""
    try:
        value = loader(tenant_id, key)
    except TypeError:
        try:
            value = loader(tenant_id=tenant_id, key=key)
        except Exception:
            return ""
    except Exception:
        return ""
    return str(value or "")


def _runtime_credentials_fingerprint(repo: Any, tenant_id: str) -> dict[str, Any]:
    loader = getattr(repo, "latest_runtime_credentials", None)
    if not callable(loader):
        return {}
    try:
        row = loader(tenant_id=tenant_id) or {}
    except Exception:
        return {}
    if not isinstance(row, Mapping):
        return {}
    fields = (
        "updated_at",
        "model_secret_name",
        "kis_secret_name",
        "kis_account_no_masked",
        "kis_env",
        "has_openai",
        "has_gemini",
        "has_anthropic",
    )
    return {field: _fingerprint_value(field, row.get(field)) for field in fields if field in row}


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
    model_id = normalize_chat_model_selection(provider_token, model_id)
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


def _agent_cache_fingerprint(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    provider: str,
    model_id: str,
) -> str:
    if is_dataclass(settings):
        settings_items = ((field.name, getattr(settings, field.name)) for field in fields(settings))
    else:
        settings_items = vars(settings).items()
    settings_payload = {
        key: _fingerprint_value(key, value)
        for key, value in sorted(settings_items, key=lambda pair: str(pair[0]))
    }
    payload = {
        "settings": settings_payload,
        "tenant_id": tenant_id,
        "provider": provider,
        "model_id": model_id,
        "selected_provider_api_key": _secret_digest(provider_api_key_from_settings(settings, provider)),
        "selected_provider_base_url": provider_base_url_from_settings(settings, provider),
        "selected_provider_has_credentials": provider_has_credentials(settings, provider),
        "runtime_credentials": _runtime_credentials_fingerprint(repo, tenant_id),
        "disabled_tools": _repo_config_value(repo, tenant_id, "disabled_tools"),
        "investment_chat_config": _repo_config_value(repo, tenant_id, "investment_chat_config"),
        "investment_chat_account_markets": _repo_config_value(
            repo,
            tenant_id,
            "investment_chat_account_markets",
        ),
        "vertex_env": {
            "GOOGLE_GENAI_USE_VERTEXAI": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", ""),
            "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            "GOOGLE_CLOUD_LOCATION": os.getenv("GOOGLE_CLOUD_LOCATION", ""),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


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
        invalidate_tenant_cache: Callable[..., Any] | None = None,
        read_only: bool = False,
    ) -> None:
        self.repo = repo
        self.settings_for_tenant = settings_for_tenant
        self.get_default_registry = get_default_registry
        self.default_tenant = normalize_tenant(default_tenant)
        self.invalidate_tenant_cache = invalidate_tenant_cache
        self.read_only = bool(read_only)
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
        allowed_providers, credential_scoped = tenant_available_provider_ids(self.repo, tenant_id=tenant)
        settings = self.settings_for_tenant(tenant)
        chat_config = load_chat_agent_config(self.repo, tenant_id=tenant)
        stored_provider = canonical_provider(chat_config.get("provider")) or str(chat_config.get("provider") or "").strip().lower()
        tenant_provider, tenant_model = tenant_default_chat_selection(settings, allowed_providers=allowed_providers)
        preferred_provider = str(
            requested_provider
            or encoded_provider
            or stored_provider
            or tenant_provider
            or "gemini"
        ).strip().lower() or "gemini"
        provider = preferred_provider
        if allowed_providers and provider not in allowed_providers:
            provider = tenant_provider if tenant_provider in allowed_providers else allowed_providers[0]
        elif credential_scoped and not allowed_providers:
            return "", ""
        preferred_model = requested_model or encoded_model
        preferred_model_provider = requested_provider or encoded_provider
        model = str(preferred_model if provider == preferred_model_provider else "").strip()
        model = normalize_chat_model_selection(provider, model)
        if not model and provider == stored_provider:
            advisor_default = tenant_model if provider == tenant_provider else default_model_for_provider(settings, provider)
            model = normalize_stored_advisor_model_selection(
                provider,
                chat_config.get("model"),
                advisor_default_model=advisor_default,
                chat_config=chat_config,
            )
        if not model and provider == tenant_provider:
            model = tenant_model
        if not model:
            model = default_model_for_provider(settings, provider)
        model = normalize_chat_model_selection(provider, model)
        return provider, model

    def load_agent(self, agent_name: str):
        if agent_name != APP_NAME and not _tenant_from_app_name(agent_name):
            raise ValueError(f"unknown investment chat app: {agent_name}")
        tenant = self._tenant_id(agent_name)
        provider, model_id = self._selection(tenant, agent_name)
        if not provider or not model_id:
            raise ValueError(f"no registered investment chat model provider for tenant: {tenant}")
        settings = self.settings_for_tenant(tenant)
        fingerprint = _agent_cache_fingerprint(
            repo=self.repo,
            settings=settings,
            tenant_id=tenant,
            provider=provider,
            model_id=model_id,
        )
        cache_key = f"{agent_name}:{tenant}:{provider}:{model_id}:{int(self.read_only)}:{fingerprint}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
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
            invalidate_tenant_cache=self.invalidate_tenant_cache,
            read_only=self.read_only,
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


_MOBILE_OVERRIDE_OPEN = "<!-- arena-mobile-overrides:start -->"
_MOBILE_OVERRIDE_CLOSE = "<!-- arena-mobile-overrides:end -->"
_MOBILE_OVERRIDE_CSS = """\
<style>
@media (max-width: 767px) {
  /* fit: release desktop min-widths */
  .chat-card { min-width: 0 !important; }
  .callback-form { min-width: 0 !important; }
  .selector-drawer.match-side-panel-width,
  side-panel-width { width: min(100vw, 92vw) !important; max-width: 100vw !important; }
  .eval-compare-container .actual-result,
  .eval-compare-container .expected-result { min-width: 0 !important; max-width: 100% !important; }
  .chat-input-container { padding: 12px 12px 16px !important; }
  .chat-messages { padding: 12px !important; }
  html, body { overflow-x: hidden !important; }

  /* messages: contain overflow from markdown tables / long tokens */
  .message-card,
  .message-text,
  .message-content { max-width: 100% !important; min-width: 0 !important; }
  .message-text { overflow-wrap: anywhere !important; word-break: break-word !important; }
  .message-text table {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
  }
  .message-text th,
  .message-text td {
    padding: 4px 8px !important;
    white-space: nowrap;
  }
  .message-text pre {
    max-width: 100% !important;
    overflow-x: auto !important;
  }
  .message-text img,
  .message-text video,
  .message-text canvas { max-width: 100% !important; height: auto !important; }

  /* toolbar: prevent icon overlap on the right (theme toggle + user avatar)
     covers both chat-page toolbar and any nested side-panel toolbar that
     re-uses the same components */
  .toolbar { padding: 0 8px !important; gap: 4px !important; flex-wrap: nowrap !important; }
  app-toolbar { padding: 0 6px !important; flex-wrap: nowrap !important; }
  .toolbar-actions, .toolbar-group {
    gap: 6px !important;
    flex-wrap: nowrap !important;
    flex-shrink: 0 !important;
  }
  user-avatar-button, .user-avatar-button {
    margin-left: 0 !important;
    flex-shrink: 0 !important;
  }
  theme-toggle-button, .theme-toggle-button {
    margin-right: 0 !important;
    margin-left: 0 !important;
    flex-shrink: 0 !important;
    width: 32px !important;
    height: 32px !important;
  }
  user-avatar, .user-avatar {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    font-size: 13px !important;
    flex-shrink: 0 !important;
  }
  user-avatar-button mat-icon, .user-avatar-button mat-icon {
    font-size: 20px !important;
    width: 20px !important;
    height: 20px !important;
  }
  theme-toggle-button mat-icon, .theme-toggle-button mat-icon {
    font-size: 18px !important;
    width: 18px !important;
    height: 18px !important;
  }

  /* drawer content: prevent right-side clipping inside the hamburger drawer */
  selector-drawer, selector-drawer .mat-drawer-inner-container {
    overflow-x: hidden !important;
    box-sizing: border-box !important;
  }
  selector-drawer-header {
    padding: 8px 8px 8px 12px !important;
    gap: 8px !important;
    min-width: 0 !important;
  }
  selector-drawer-title {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
  }
  app-selector-search { padding: 0 8px 4px !important; }
  app-selector-list { padding: 0 6px !important; }
  app-selector-item {
    padding: 10px 8px !important;
    gap: 8px !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
  }
  app-selector-item-name {
    min-width: 0 !important;
    flex: 1 1 auto !important;
  }
  session-info {
    padding: 8px !important;
    min-width: 0 !important;
    overflow: hidden !important;
  }
  session-info .session-header { gap: 4px !important; min-width: 0 !important; }
  session-info .session-header .session-id { min-width: 0 !important; }

  /* chat input: fill viewport on mobile (was capped to 88%) */
  .chat-input {
    width: 100% !important;
    padding: 8px 10px !important;
    box-sizing: border-box !important;
  }
  .chat-input-content-row { gap: 8px !important; }
  .chat-input-actions { margin-top: 8px !important; gap: 4px !important; }

  /* iOS auto-zoom prevention: text inputs need >=16px font-size */
  input-box,
  .chat-input-box,
  textarea.chat-input-box,
  .mat-mdc-form-field input,
  .mat-mdc-form-field textarea,
  app-selector-search-field input {
    font-size: 16px !important;
  }
  input-box { line-height: 22px !important; }

  /* toolbar selector buttons: cap tighter so user/theme stay reachable */
  selector-button {
    max-width: min(160px, 35vw) !important;
    padding: 4px 8px !important;
  }

  /* trace row: release the desktop-only fixed left column */
  trace-row-left {
    min-width: 0 !important;
    width: 40% !important;
    max-width: 50% !important;
  }

  /* hover-only message action buttons: always visible on touch */
  .message-card:hover button,
  .message-card button,
  .message-feedback-container button { visibility: visible !important; }

  /* momentum scroll on touch */
  .chat-messages,
  .mat-drawer-inner-container,
  app-selector-list,
  selector-drawer-content,
  .mat-mdc-dialog-content,
  .message-text pre,
  .message-text table { -webkit-overflow-scrolling: touch !important; }

  /* dialog: maximize content area on small screens */
  .mat-mdc-dialog-panel {
    width: calc(100vw - 16px) !important;
    max-width: calc(100vw - 16px) !important;
  }
  .mat-mdc-dialog-content { padding: 12px 16px !important; }

  /* subtle tap highlight (avoid blue flash on every tap) */
  button, [role="button"], a, mat-list-item, mat-option {
    -webkit-tap-highlight-color: rgba(0, 0, 0, .04);
  }

  /* safe-area inset (iOS notch / home indicator) on bottom-anchored UI */
  .chat-input-container {
    padding-bottom: max(16px, env(safe-area-inset-bottom)) !important;
  }
}
</style>
"""


def _inject_mobile_overrides(index_html_path: Path) -> None:
    if not index_html_path.exists():
        return
    text = index_html_path.read_text(encoding="utf-8")
    block = f"{_MOBILE_OVERRIDE_OPEN}\n{_MOBILE_OVERRIDE_CSS}{_MOBILE_OVERRIDE_CLOSE}\n"
    if text.count(_MOBILE_OVERRIDE_OPEN) == 1 and block in text:
        return
    while True:
        start = text.find(_MOBILE_OVERRIDE_OPEN)
        end = text.find(_MOBILE_OVERRIDE_CLOSE)
        if start == -1 or end == -1 or end <= start:
            break
        text = text[:start] + text[end + len(_MOBILE_OVERRIDE_CLOSE):].lstrip("\n")
    head_close = text.lower().find("</head>")
    if head_close < 0:
        return
    patched = text[:head_close] + block + text[head_close:]
    index_html_path.write_text(patched, encoding="utf-8")


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
    _inject_mobile_overrides(dest / "index.html")
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
    if os.getenv("K_SERVICE"):
        logger.warning(
            "ARENA_CHAT_SESSION_SERVICE_URI is not set on Cloud Run; "
            "investment chat ADK sessions will use an ephemeral sqlite database at %s",
            session_db_path,
        )
    return f"sqlite:///{session_db_path}"


def _app_name_from_path(path: str) -> str:
    match = re.search(r"/apps/([^/]+)", str(path or ""))
    return unquote(match.group(1)) if match else ""


async def _app_name_from_request(request: Request) -> str:
    app_name = _app_name_from_path(str(request.url.path or ""))
    if app_name:
        return app_name
    path = str(request.url.path or "").rstrip("/")
    if str(getattr(request, "method", "") or "").upper() not in {"POST", "PATCH"}:
        return ""
    if not path.endswith(("/run", "/run_sse")):
        return ""
    try:
        body = await request.body()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("app_name") or "").strip()


async def _stale_app_name_response(
    request: Request,
    *,
    tenant: str,
    provider: str,
    model: str,
) -> JSONResponse | None:
    app_name = await _app_name_from_request(request)
    if not app_name:
        return None
    app_tenant, app_provider, app_model = _spec_from_app_name(app_name)
    if app_tenant and app_tenant != tenant:
        return JSONResponse(
            {
                "error": "stale adk app_name tenant",
                "tenant_id": tenant,
                "app_name": app_name,
                "app_name_tenant": app_tenant,
            },
            status_code=409,
        )
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    app_provider_token = canonical_provider(app_provider) or app_provider
    if app_provider_token and provider_token and app_provider_token != provider_token:
        return JSONResponse(
            {
                "error": "stale adk app_name provider",
                "tenant_id": tenant,
                "provider": provider_token,
                "app_name": app_name,
                "app_name_provider": app_provider_token,
            },
            status_code=409,
        )
    model_token = normalize_chat_model_selection(provider_token or app_provider_token, model)
    app_model_token = normalize_chat_model_selection(app_provider_token or provider_token, app_model)
    if app_model_token and model_token and app_model_token != model_token:
        return JSONResponse(
            {
                "error": "stale adk app_name model",
                "tenant_id": tenant,
                "model": model_token,
                "app_name": app_name,
                "app_name_model": app_model_token,
            },
            status_code=409,
        )
    return None


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
        query_provider = str(request.query_params.get("provider") or "").strip().lower()
        query_model = str(request.query_params.get("model") or "").strip()
        if query_provider:
            provider = canonical_provider(query_provider) or query_provider
            try:
                request.session["investment_chat_provider"] = provider
            except Exception:
                pass
        if query_model:
            model = normalize_chat_model_selection(provider, query_model)
            try:
                request.session["investment_chat_model"] = model
            except Exception:
                pass
        provider = canonical_provider(provider) or str(provider or "").strip().lower()
        model = normalize_chat_model_selection(provider, model)
        tenant_token = REQUEST_TENANT.set(tenant)
        user_token = REQUEST_USER_EMAIL.set(user_email)
        provider_token = REQUEST_PROVIDER.set(provider)
        model_token = REQUEST_MODEL.set(model)
        try:
            if auth_enabled and not user:
                accept = str(request.headers.get("accept") or "").lower()
                path = str(request.url.path or "")
                if "text/html" in accept or "/dev-ui" in path:
                    return RedirectResponse("/auth/google/login", status_code=302)
                return JSONResponse({"error": "auth required"}, status_code=401)

            stale_response = await _stale_app_name_response(
                request,
                tenant=tenant,
                provider=provider,
                model=model,
            )
            if stale_response is not None:
                return stale_response
            return await call_next(request)
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
    invalidate_tenant_cache: Callable[..., Any] | None = None,
    read_only: bool = False,
) -> FastAPI:
    loader = InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=settings_for_tenant,
        get_default_registry=get_default_registry,
        default_tenant=default_tenant,
        invalidate_tenant_cache=invalidate_tenant_cache,
        read_only=read_only,
    )
    session_service_uri = _default_session_service_uri()
    artifact_service_uri = str(os.getenv("ARENA_CHAT_ARTIFACT_SERVICE_URI") or "memory://").strip() or "memory://"
    with _SuppressAdkExperimentalWarnings():
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
