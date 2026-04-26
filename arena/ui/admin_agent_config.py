"""Admin agent config domain logic.

Pure parsing/normalization/serialization plus a thin repo-bound store for
reading and writing the per-tenant ``agents_config`` payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from arena.config import Settings
from arena.providers import canonical_provider


_MARKET_ALIASES = {"kr": "kospi", "korea": "kospi"}
_ALLOWED_MARKETS = {"us", "nasdaq", "nyse", "amex", "kospi", "kosdaq"}
_US_SUBMARKETS = {"nasdaq", "nyse", "amex"}


def normalize_market_tokens(raw_market: object) -> list[str]:
    tokens: list[str] = []
    for token in str(raw_market or "").split(","):
        market = _MARKET_ALIASES.get(str(token).strip().lower(), str(token).strip().lower())
        if not market or market not in _ALLOWED_MARKETS or market in tokens:
            continue
        tokens.append(market)
    if "us" in tokens:
        tokens = [t for t in tokens if t == "us" or t not in _US_SUBMARKETS]
    return tokens


def derive_tenant_market_from_agents(
    entries: list[dict[str, Any]],
    *,
    fallback_market: str,
) -> str:
    tokens: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for market in normalize_market_tokens(entry.get("target_market")):
            if market not in tokens:
                tokens.append(market)
    if not tokens:
        tokens = normalize_market_tokens(fallback_market)
    return ",".join(tokens)


def serialize_agents_config_entries(
    entries: list[dict[str, Any]],
    *,
    tenant_settings: Settings,
    safe_float: Callable[[object, float], float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("id") or "").strip().lower()
        if not aid:
            continue
        capital = safe_float(entry.get("capital_krw"), tenant_settings.sleeve_capital_krw)
        if capital <= 0:
            capital = tenant_settings.sleeve_capital_krw
        db_entry: dict[str, Any] = {
            "id": aid,
            "provider": str(entry.get("provider") or "").strip().lower(),
            "model": str(entry.get("model") or "").strip(),
            "capital_krw": capital,
        }
        if entry.get("target_market"):
            db_entry["target_market"] = str(entry["target_market"]).strip().lower()
        if entry.get("system_prompt"):
            db_entry["system_prompt"] = str(entry["system_prompt"]).strip()
        if entry.get("risk_policy"):
            db_entry["risk_policy"] = entry["risk_policy"]
        if isinstance(entry.get("disabled_tools"), list):
            db_entry["disabled_tools"] = [str(x).strip() for x in entry["disabled_tools"] if str(x).strip()]
        if isinstance(entry.get("llm_params"), dict) and entry["llm_params"]:
            db_entry["llm_params"] = entry["llm_params"]
        memory_model = str(entry.get("memory_compaction_model") or "").strip()
        if memory_model:
            db_entry["memory_compaction_model"] = memory_model
        out.append(db_entry)
    return out


def build_single_agent_entry(
    *,
    agent_data: dict[str, Any],
    existing_entry: dict[str, Any] | None,
    tenant_settings: Settings,
    allowed_providers: set[str],
    provider_aliases: dict[str, str],
    safe_float: Callable[[object, float], float],
) -> tuple[str, str, dict[str, Any], str]:
    current = dict(existing_entry or {})
    aid = str(agent_data.get("id") or agent_data.get("agent_id") or current.get("id") or "").strip().lower()
    if not aid:
        raise ValueError("agent.id is required")

    provider_raw = agent_data.get("provider") if "provider" in agent_data else current.get("provider")
    provider = canonical_provider(str(provider_raw or "").strip().lower())
    if not provider:
        provider = provider_aliases.get(aid, "")
    if provider not in allowed_providers:
        raise ValueError(f"invalid provider: {provider}")

    model = str(agent_data.get("model") if "model" in agent_data else current.get("model") or "").strip()
    capital_raw = agent_data.get("capital_krw") if "capital_krw" in agent_data else current.get("capital_krw")
    capital = safe_float(capital_raw, tenant_settings.sleeve_capital_krw)
    if capital <= 0:
        capital = safe_float(current.get("capital_krw"), tenant_settings.sleeve_capital_krw)
    if capital <= 0:
        capital = tenant_settings.sleeve_capital_krw

    target_market = (
        str(agent_data.get("target_market") if "target_market" in agent_data else current.get("target_market") or "")
        .strip()
        .lower()
    )
    system_prompt = str(
        agent_data.get("system_prompt") if "system_prompt" in agent_data else current.get("system_prompt") or ""
    ).strip()
    risk_policy = agent_data.get("risk_policy") if "risk_policy" in agent_data else current.get("risk_policy")
    disabled_tools_raw = (
        agent_data.get("disabled_tools") if "disabled_tools" in agent_data else current.get("disabled_tools")
    )
    llm_params_raw = agent_data.get("llm_params") if "llm_params" in agent_data else current.get("llm_params")
    memory_compaction_model = str(
        agent_data.get("memory_compaction_model")
        if "memory_compaction_model" in agent_data
        else current.get("memory_compaction_model") or ""
    ).strip()

    entry: dict[str, Any] = {
        "id": aid,
        "provider": provider,
        "model": model,
        "capital_krw": capital,
    }
    if target_market:
        entry["target_market"] = target_market
    if system_prompt:
        entry["system_prompt"] = system_prompt
    if isinstance(risk_policy, dict) and risk_policy:
        entry["risk_policy"] = risk_policy
    if isinstance(disabled_tools_raw, list):
        entry["disabled_tools"] = [str(x).strip() for x in disabled_tools_raw if str(x).strip()]
    if isinstance(llm_params_raw, dict) and llm_params_raw:
        from arena.agents.llm_params import sanitize_llm_params

        cleaned = sanitize_llm_params(provider, llm_params_raw)
        if cleaned:
            entry["llm_params"] = cleaned
    if memory_compaction_model:
        entry["memory_compaction_model"] = memory_compaction_model

    raw_api_key = str(agent_data.get("api_key") or "").strip()
    return aid, provider, entry, raw_api_key


@dataclass(frozen=True)
class AgentsConfigSavePayload:
    """Normalized batch save payload for the admin agents route."""

    entries: list[dict[str, Any]]
    api_keys: dict[str, dict[str, str]]

    @property
    def agent_ids(self) -> list[str]:
        return [str(entry.get("id") or "").strip().lower() for entry in self.entries]

    @property
    def models_by_provider(self) -> dict[str, str]:
        models: dict[str, str] = {}
        for entry in self.entries:
            provider = str(entry.get("provider") or "")
            model = str(entry.get("model") or "")
            if provider and model and provider not in models:
                models[provider] = model
        return models


def build_agents_config_save_payload(
    raw_entries: list[Any],
    *,
    tenant_settings: Settings,
    allowed_providers: set[str],
    provider_aliases: dict[str, str],
    safe_float: Callable[[object, float], float],
) -> AgentsConfigSavePayload:
    entries: list[dict[str, Any]] = []
    api_keys: dict[str, dict[str, str]] = {}

    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        try:
            _aid, provider, entry, raw_api_key = build_single_agent_entry(
                agent_data=item,
                existing_entry=None,
                tenant_settings=tenant_settings,
                allowed_providers=allowed_providers,
                provider_aliases=provider_aliases,
                safe_float=safe_float,
            )
        except ValueError:
            continue
        entries.append(entry)

        provider_payload = dict(api_keys.get(provider) or {})
        if raw_api_key:
            provider_payload["api_key"] = raw_api_key
        model = str(entry.get("model") or "").strip()
        if model:
            provider_payload["model"] = model
        if provider_payload:
            api_keys[provider] = provider_payload

    return AgentsConfigSavePayload(entries=entries, api_keys=api_keys)


@dataclass(frozen=True)
class AdminAgentConfigStore:
    """Reads/writes agents_config and the derived ``kis_target_market`` config."""

    repo: Any
    current_admin_view_model: Callable[[str], dict[str, Any]]

    def load_explicit_rows(self, tenant: str) -> tuple[list[dict[str, Any]], bool]:
        raw_loader = getattr(self.repo, "get_config", None)
        if not callable(raw_loader):
            return [], False
        try:
            raw_value = raw_loader(tenant, "agents_config")
        except Exception:
            raw_value = None
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            return [], False
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return [], False
        if not isinstance(parsed, list):
            return [], False
        return [dict(entry) for entry in parsed if isinstance(entry, dict)], True

    def load_for_update(self, tenant: str) -> tuple[list[dict[str, Any]], bool]:
        entries, has_explicit = self.load_explicit_rows(tenant)
        if has_explicit:
            return entries, True
        vm = self.current_admin_view_model(tenant)
        fallback = [
            dict(entry)
            for entry in list(vm.get("agents_config") or [])
            if isinstance(entry, dict)
        ]
        return fallback, False

    def sync_market(
        self,
        *,
        tenant: str,
        entries: list[dict[str, Any]],
        tenant_settings: Settings,
        updated_by: str,
    ) -> str:
        current_market = getattr(tenant_settings, "kis_target_market", "").strip().lower()
        next_market = derive_tenant_market_from_agents(entries, fallback_market=current_market)
        if next_market and next_market != current_market:
            self.repo.set_config(tenant, "kis_target_market", next_market, updated_by)
        return next_market or current_market
