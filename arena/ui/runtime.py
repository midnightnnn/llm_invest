from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable

from arena.cli_runtime import _apply_shared_research_gemini, _apply_tenant_runtime_credentials
from arena.config import Settings, apply_runtime_overrides, research_generation_status
from arena.data.bq import BigQueryRepository
from arena.providers import (
    default_model_for_provider,
    list_adk_provider_specs,
    runtime_row_api_key_status,
)
from arena.tools.default_registry import build_default_registry

logger = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool_token(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    token = str(value).strip().lower()
    if not token:
        return default
    return token in {"1", "true", "yes", "y", "on"}


def _parse_json_array_state(raw: object, *, field_name: str) -> tuple[list[Any], bool]:
    text = str(raw or "").strip()
    if not text:
        return [], False
    try:
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(
            "[yellow]UI runtime JSON parse failed[/yellow] field=%s err=%s",
            field_name,
            str(exc),
        )
        return [], True
    if isinstance(parsed, list):
        return parsed, False
    logger.warning(
        "[yellow]UI runtime JSON type mismatch[/yellow] field=%s expected=array actual=%s",
        field_name,
        type(parsed).__name__,
    )
    return [], True


def _parse_json_array(raw: object) -> list[Any]:
    parsed, _invalid = _parse_json_array_state(raw, field_name="json_array")
    return parsed


def _parse_json_object_state(raw: object, *, field_name: str) -> tuple[dict[str, Any], bool]:
    text = str(raw or "").strip()
    if not text:
        return {}, False
    try:
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(
            "[yellow]UI runtime JSON parse failed[/yellow] field=%s err=%s",
            field_name,
            str(exc),
        )
        return {}, True
    if isinstance(parsed, dict):
        return parsed, False
    logger.warning(
        "[yellow]UI runtime JSON type mismatch[/yellow] field=%s expected=object actual=%s",
        field_name,
        type(parsed).__name__,
    )
    return {}, True


def _parse_json_object(raw: object) -> dict[str, Any]:
    parsed, _invalid = _parse_json_object_state(raw, field_name="json_object")
    return parsed


def _mask_account_display(account_no: str) -> str:
    digits = re.sub(r"\D", "", str(account_no or ""))
    if len(digits) < 4:
        return ""
    return f"{'*' * max(0, len(digits) - 4)}{digits[-4:]}"


class UIRuntime:
    def __init__(
        self,
        *,
        repo: BigQueryRepository,
        settings: Settings,
        executor: ThreadPoolExecutor,
        default_prompt_template: Callable[[str], str],
        credential_store: Any | None = None,
    ) -> None:
        self.repo = repo
        self.settings = settings
        self.executor = executor
        self.default_prompt_template = default_prompt_template
        self.credential_store = credential_store
        self._query_cache: dict[str, tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl_default = 90.0
        self._cache_ttl_map: dict[str, float] = {
            "runtime_config:": 300.0,
            "tenant_settings:": 300.0,
            "registry:": 600.0,
            "tenant_run_status:": 60.0,
            "nav:": 120.0,
            "board:": 60.0,
            "trades:": 60.0,
            "sleeves:": 120.0,
            "sleeve_cards:": 120.0,
            "settings:": 120.0,
            "memory_stats:": 120.0,
        }

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._query_cache.clear()

    def _ttl_for_key(self, cache_key: str) -> float:
        for prefix, ttl in self._cache_ttl_map.items():
            if cache_key.startswith(prefix):
                return ttl
        return self._cache_ttl_default

    def cached_fetch(self, cache_key: str, fetch_fn: Callable[..., Any], *args, **kwargs):
        now = time.monotonic()
        ttl = self._ttl_for_key(cache_key)
        with self._cache_lock:
            entry = self._query_cache.get(cache_key)
            if entry and (now - entry[0]) < ttl:
                return entry[1]
        result = fetch_fn(*args, **kwargs)
        with self._cache_lock:
            self._query_cache[cache_key] = (time.monotonic(), result)
        return result

    def invalidate_prefix(self, prefix: str) -> None:
        token = str(prefix or "")
        if not token:
            return
        with self._cache_lock:
            stale_keys = [key for key in self._query_cache if str(key).startswith(token)]
            for key in stale_keys:
                self._query_cache.pop(key, None)

    def invalidate_tenant_cache(self, tenant: str, *scopes: str) -> None:
        token = str(tenant or "").strip().lower() or "local"
        active_scopes = tuple(str(scope or "").strip().lower() for scope in scopes if str(scope or "").strip()) or (
            "runtime",
        )
        for scope in active_scopes:
            if scope == "runtime":
                self.invalidate_prefix(f"runtime_config:{token}")
                self.invalidate_prefix(f"tenant_settings:{token}")
                self.invalidate_prefix(f"registry:{token}:")
            elif scope == "memory":
                self.invalidate_prefix(f"memory_stats:{token}:")
            elif scope == "portfolio":
                self.invalidate_prefix(f"nav:{token}:")
                self.invalidate_prefix(f"sleeves:{token}:")
                self.invalidate_prefix(f"sleeve_cards:{token}:")
            elif scope == "status":
                self.invalidate_prefix(f"tenant_run_status:{token}")

    def invalidate_tenant_runtime_cache(self, tenant: str) -> None:
        self.invalidate_tenant_cache(tenant, "runtime", "memory")

    def read_runtime_config(self, tenant: str) -> dict[str, str]:
        keys = [
            "system_prompt",
            "risk_policy",
            "sleeve_capital_krw",
            "disabled_tools",
            "mcp_servers",
            "agents_config",
            "universe_run_top_n",
            "universe_per_exchange_cap",
            "forecast_mode",
            "reddit_sentiment_enabled",
            "research_max_tickers",
            "research_mover_top_n",
            "research_earnings_lookahead_days",
        ]
        values: dict[str, str] = {}
        try:
            values = dict(self.repo.get_configs(tenant, keys) or {})
        except Exception:
            values = {}
        if len(values) < len(keys):
            for key in keys:
                if key in values:
                    continue
                try:
                    val = self.repo.get_config(tenant, key)
                except Exception:
                    val = None
                if val is not None:
                    values[key] = str(val)
        return values

    def settings_for_tenant(self, tenant: str) -> Settings:
        tenant_norm = str(tenant or "").strip().lower() or "local"

        def _build() -> Settings:
            tenant_settings = deepcopy(self.settings)
            _apply_tenant_runtime_credentials(tenant_settings, self.repo, tenant_id=tenant_norm)
            tenant_settings = apply_runtime_overrides(tenant_settings, self.repo, tenant_id=tenant_norm)
            _apply_shared_research_gemini(tenant_settings, self.repo, tenant_id=tenant_norm)
            return tenant_settings

        return self.cached_fetch(f"tenant_settings:{tenant_norm}", _build)

    def get_default_registry(self, tenant: str):
        tenant_norm = str(tenant or "").strip().lower() or "local"
        tenant_settings = self.settings_for_tenant(tenant_norm)
        try:
            tools_config_raw = str(self.repo.get_config(tenant_norm, "tools_config") or "")
        except Exception:
            tools_config_raw = ""
        registry_sig = (
            int(bool(getattr(tenant_settings, "reddit_sentiment_enabled", False))),
            hash(tools_config_raw),
        )
        return self.cached_fetch(
            f"registry:{tenant_norm}:{registry_sig}",
            build_default_registry,
            repo=self.repo,
            settings=tenant_settings,
            tenant_id=tenant_norm,
        )

    def current_admin_view_model(self, tenant: str, *, latest_creds: dict[str, Any] | None = None) -> dict[str, Any]:
        settings_fut = self.executor.submit(self.settings_for_tenant, tenant)
        cfg_fut = self.executor.submit(self.cached_fetch, f"runtime_config:{tenant}", self.read_runtime_config, tenant)
        registry_fut = self.executor.submit(self.get_default_registry, tenant)

        tenant_settings = settings_fut.result()
        cfg = cfg_fut.result()
        registry = registry_fut.result()

        core_prompt_text = self.default_prompt_template("core_prompt.txt")
        prompt_text = str(cfg.get("system_prompt") or "").strip() or self.default_prompt_template("system_prompt.txt")

        agent_ids = [str(x).strip().lower() for x in tenant_settings.agent_ids if str(x).strip()]

        invalid_runtime_config_keys: list[str] = []

        risk_raw, risk_invalid = _parse_json_object_state(cfg.get("risk_policy"), field_name="risk_policy")
        if risk_invalid:
            invalid_runtime_config_keys.append("risk_policy")
        risk = {
            "max_order_krw": _safe_float(risk_raw.get("max_order_krw"), tenant_settings.max_order_krw),
            "max_daily_turnover_ratio": _safe_float(risk_raw.get("max_daily_turnover_ratio"), tenant_settings.max_daily_turnover_ratio),
            "max_position_ratio": _safe_float(risk_raw.get("max_position_ratio"), tenant_settings.max_position_ratio),
            "min_cash_buffer_ratio": _safe_float(risk_raw.get("min_cash_buffer_ratio"), tenant_settings.min_cash_buffer_ratio),
            "ticker_cooldown_seconds": _safe_int(risk_raw.get("ticker_cooldown_seconds"), tenant_settings.ticker_cooldown_seconds),
            "max_daily_orders": _safe_int(risk_raw.get("max_daily_orders"), tenant_settings.max_daily_orders),
            "estimated_fee_bps": _safe_float(risk_raw.get("estimated_fee_bps"), tenant_settings.estimated_fee_bps),
        }

        sleeve_capital_krw = float(tenant_settings.sleeve_capital_krw)

        disabled_tools_values, disabled_tools_invalid = _parse_json_array_state(
            cfg.get("disabled_tools"),
            field_name="disabled_tools",
        )
        if disabled_tools_invalid:
            invalid_runtime_config_keys.append("disabled_tools")
        disabled_tools_raw = sorted(
            {
                str(x).strip()
                for x in disabled_tools_values
                if str(x).strip()
            }
        )
        mcp_server_values, mcp_servers_invalid = _parse_json_array_state(
            cfg.get("mcp_servers"),
            field_name="mcp_servers",
        )
        if mcp_servers_invalid:
            invalid_runtime_config_keys.append("mcp_servers")
        mcp_servers: list[dict[str, Any]] = []
        for item in mcp_server_values:
            if isinstance(item, dict):
                mcp_servers.append(
                    {
                        "name": str(item.get("name") or "").strip(),
                        "url": str(item.get("url") or "").strip(),
                        "transport": str(item.get("transport") or "sse").strip().lower() or "sse",
                        "enabled": _to_bool_token(item.get("enabled"), True),
                    }
                )
        import inspect as _inspect

        # Build a lookup for core tools from _ContextTools class
        _ctx_methods: dict[str, Any] = {}
        try:
            from arena.agents.adk_context_tools import _ContextTools
            for attr_name in dir(_ContextTools):
                if attr_name.startswith("_"):
                    continue
                method = getattr(_ContextTools, attr_name, None)
                if callable(method):
                    _ctx_methods[attr_name] = method
        except Exception:
            pass

        def _introspect(fn):
            params = []
            source = ""
            try:
                sig = _inspect.signature(fn)
                for pname, p in sig.parameters.items():
                    if pname in ("self", "cls"):
                        continue
                    ann = p.annotation if p.annotation != _inspect.Parameter.empty else None
                    default = p.default if p.default != _inspect.Parameter.empty else None
                    params.append({
                        "name": pname,
                        "type": str(ann) if ann else "",
                        "default": repr(default) if default is not None else "",
                        "required": p.default is _inspect.Parameter.empty,
                    })
            except (ValueError, TypeError):
                pass
            try:
                source = _inspect.getsource(fn)
            except (OSError, TypeError):
                pass
            return params, source

        def _tool_detail(e):
            entry = {
                "tool_id": e.tool_id,
                "name": e.name,
                "label_ko": e.label_ko or e.tool_id,
                "description": e.description,
                "description_ko": e.description_ko or e.description,
                "category": e.category,
                "tier": e.tier,
                "configurable": True,
                "params": [],
                "source": "",
            }
            fn = e.callable
            if fn is None:
                fn = _ctx_methods.get(e.tool_id)
            if fn is not None:
                entry["params"], entry["source"] = _introspect(fn)
            return entry

        tool_entries = [_tool_detail(e) for e in registry.list_entries()]
        allowed_tool_ids = {str(e.get("tool_id") or "") for e in tool_entries}
        disabled_tools = [tool_id for tool_id in disabled_tools_raw if tool_id in allowed_tool_ids]
        agents_config: list[dict[str, Any]] = []
        active_agent_ids: list[str] = []
        for aid in agent_ids:
            ac = tenant_settings.agent_configs.get(aid)
            if ac is None:
                continue
            active_agent_ids.append(aid)
            ac_entry: dict[str, Any] = {
                "id": aid,
                "provider": str(ac.provider or "").strip().lower(),
                "model": str(ac.model or "").strip(),
                "capital_krw": _safe_float(ac.capital_krw, sleeve_capital_krw),
            }
            if ac.system_prompt:
                ac_entry["system_prompt"] = str(ac.system_prompt).strip()
            if isinstance(ac.risk_overrides, dict) and ac.risk_overrides:
                ac_entry["risk_policy"] = dict(ac.risk_overrides)
            if isinstance(ac.disabled_tools, list):
                ac_entry["disabled_tools"] = [str(x).strip() for x in ac.disabled_tools if str(x).strip()]
            if ac.target_market:
                ac_entry["target_market"] = str(ac.target_market).strip().lower()
            agents_config.append(ac_entry)
        if active_agent_ids:
            agent_ids = active_agent_ids

        agent_models = {
            spec.provider_id: default_model_for_provider(tenant_settings, spec.provider_id)
            for spec in list_adk_provider_specs()
        }

        api_key_status: dict[str, bool] = {}
        try:
            creds = latest_creds if latest_creds is not None else (self.repo.latest_runtime_credentials(tenant_id=tenant) or {})
            api_key_status = runtime_row_api_key_status(creds)
        except Exception:
            api_key_status = {}
        research_status = research_generation_status(tenant_settings)

        kis_accounts_meta: list[dict[str, str]] = []
        if self.credential_store is not None:
            try:
                kis_accounts_meta = self.credential_store.list_kis_accounts_meta(tenant_id=tenant)
            except Exception:
                kis_accounts_meta = []

        active_kis_digits = re.sub(r"\D", "", str(getattr(tenant_settings, "kis_account_no", "") or ""))
        active_kis_product_code = str(getattr(tenant_settings, "kis_account_product_code", "01") or "01").strip() or "01"
        active_kis_account_no_masked = ""
        if active_kis_digits:
            if len(active_kis_digits) >= 10:
                active_kis_account_no_masked = _mask_account_display(active_kis_digits[:10])
            else:
                active_kis_account_no_masked = _mask_account_display(active_kis_digits)

        return {
            "core_prompt_text": core_prompt_text,
            "prompt_text": prompt_text,
            "agent_ids": agent_ids,
            "agent_models": agent_models,
            "risk": risk,
            "sleeve_capital_krw": sleeve_capital_krw,
            "disabled_tools": disabled_tools,
            "mcp_servers": mcp_servers,
            "invalid_runtime_config_keys": invalid_runtime_config_keys,
            "tool_entries": tool_entries,
            "agents_config": agents_config,
            "api_key_status": api_key_status,
            "research_status": research_status,
            "kis_accounts_meta": kis_accounts_meta,
            "active_kis_account_no": active_kis_digits[:10] if active_kis_digits else "",
            "active_kis_product_code": active_kis_product_code,
            "active_kis_account_no_masked": active_kis_account_no_masked,
            "tenant_settings": tenant_settings,
        }
