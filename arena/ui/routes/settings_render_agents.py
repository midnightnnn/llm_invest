from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from arena.config import Settings
from arena.ui.templating import render_ui_template


def _build_research_banner(research_status: Mapping[str, Any]) -> dict[str, str] | None:
    code = str(research_status.get("code") or "").strip().lower()
    message = str(research_status.get("message") or "").strip()
    if not message:
        return None
    classes = (
        "mb-4 rounded-2xl border border-emerald-200 bg-emerald-50/80 px-4 py-3 text-sm text-emerald-900"
        if code in {"enabled", "vertex_enabled", "shared_live_tenant"}
        else "mb-4 rounded-2xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-900"
    )
    detail = message
    if code == "missing_gemini_key":
        fallback_tenant = str(research_status.get("public_fallback_tenant") or "").strip()
        if fallback_tenant:
            detail = (
                "Gemini 키가 없어 새로운 리서치 브리핑 생성은 비활성화됩니다. "
                f"대신 {fallback_tenant}의 공용 글로벌/지정학/섹터 브리핑 조회는 계속 동작합니다."
            )
        else:
            detail = (
                "Gemini 키가 없어 새로운 리서치 브리핑 생성은 비활성화됩니다. "
                "이미 저장된 브리핑 조회는 계속 동작합니다."
            )
    elif code == "disabled_by_config":
        detail = "이 테넌트는 설정상 리서치 브리핑 생성을 끈 상태입니다."
    elif code == "vertex_enabled":
        if bool(research_status.get("vertex_limited_to_live_tenants")):
            source_tenant = str(research_status.get("shared_source_tenant") or "").strip()
            source_label = source_tenant or "operator tenant"
            detail = (
                f"이 테넌트는 승인된 live tenant라서 {source_label} 기준 operator-managed Vertex Gemini로 "
                "리서치 브리핑을 생성합니다."
            )
        else:
            detail = "이 테넌트는 Vertex AI를 통해 Gemini native grounding으로 리서치 브리핑을 생성합니다."
    elif code == "shared_live_tenant":
        source_tenant = str(research_status.get("research_source_tenant") or "").strip()
        source_label = source_tenant or "operator tenant"
        detail = (
            f"이 테넌트는 승인된 live tenant라서 {source_label}의 operator-managed Gemini로 "
            "리서치 브리핑을 생성합니다."
        )
    elif code == "enabled":
        detail = "이 테넌트는 Gemini native grounding으로 새로운 리서치 브리핑을 생성할 수 있습니다."
    return {"classes": classes, "detail": detail}


def _build_agent_card_context(
    *,
    entry: dict[str, Any],
    api_status: Mapping[str, Any],
    tenant_settings: Settings,
    prompt_text: str,
    risk_fields_meta: Sequence[tuple[str, str, Any]],
    provider_key_help: Mapping[str, str],
    provider_api_key_help_html: Callable[[str], str],
) -> dict[str, Any]:
    aid = str(entry["id"])
    provider = str(entry.get("provider") or "")
    model = str(entry.get("model") or "")
    try:
        capital_krw = max(float(entry.get("capital_krw") or tenant_settings.sleeve_capital_krw), 0.0)
    except (TypeError, ValueError):
        capital_krw = float(tenant_settings.sleeve_capital_krw)
    capital_value = str(int(round(capital_krw)))
    has_key = bool(api_status.get(aid) or api_status.get(provider))

    target_market_raw = str(entry.get("target_market") or "").strip().lower() or str(
        tenant_settings.kis_target_market or "us"
    )
    selected_markets = {m.strip() for m in target_market_raw.split(",") if m.strip()}
    markets = [
        {"key": key, "label": label, "selected": key in selected_markets}
        for key, label in (("us", "US"), ("kospi", "KOSPI"))
    ]

    sys_prompt = str(entry.get("system_prompt") or "")
    display_prompt = sys_prompt if sys_prompt else prompt_text

    risk_policy = entry.get("risk_policy") or {}
    risk_inputs = [
        {
            "key": risk_key,
            "label": risk_label,
            "value": str(risk_policy.get(risk_key, "")),
            "placeholder": str(risk_default),
        }
        for risk_key, risk_label, risk_default in risk_fields_meta
    ]

    return {
        "aid": aid,
        "provider": provider,
        "model": model,
        "capital_value": capital_value,
        "has_key": has_key,
        "markets": markets,
        "display_prompt": display_prompt,
        "risk_inputs": risk_inputs,
        "api_key_help": provider_key_help.get(provider, provider_api_key_help_html(provider)),
    }


def build_agents_panel(
    *,
    agents_cfg: Sequence[dict[str, Any]],
    api_status: Mapping[str, Any],
    research_status: Mapping[str, Any],
    tenant_settings: Settings,
    prompt_text: str,
    provider_options_html: str,
    provider_key_help: Mapping[str, str],
    default_models: Mapping[str, str],
    configurable_tools: Sequence[dict[str, Any]],
    risk_fields_meta: Sequence[tuple[str, str, Any]],
    tenant: str,
    user_email: str | None,
    provider_api_key_help_html: Callable[[str], str],
    is_live_mode: Callable[[Settings | None], bool],
    default_capital_krw: int,
) -> str:
    cards = [
        _build_agent_card_context(
            entry=entry,
            api_status=api_status,
            tenant_settings=tenant_settings,
            prompt_text=prompt_text,
            risk_fields_meta=risk_fields_meta,
            provider_key_help=provider_key_help,
            provider_api_key_help_html=provider_api_key_help_html,
        )
        for entry in agents_cfg
    ]

    tool_labels = {
        str(tool["tool_id"]): str(tool.get("label_ko") or tool["tool_id"])
        for tool in configurable_tools
    }
    tool_tips = {
        str(tool["tool_id"]): str(tool.get("description_ko") or tool["description"][:60])
        for tool in configurable_tools
    }

    return render_ui_template(
        "settings_agents_panel.jinja2",
        cards=cards,
        research_banner=_build_research_banner(research_status),
        provider_options_html=provider_options_html,
        tools_json=json.dumps(list(configurable_tools), ensure_ascii=False),
        risk_fields_json=json.dumps(
            [(k, l, str(d)) for k, l, d in risk_fields_meta], ensure_ascii=False
        ),
        default_models_json=json.dumps(dict(default_models)),
        provider_key_help_json=json.dumps(dict(provider_key_help), ensure_ascii=False),
        default_capital_krw=int(default_capital_krw),
        default_market=str(tenant_settings.kis_target_market or "us"),
        tenant_id_json=json.dumps(tenant),
        default_prompt_json=json.dumps(prompt_text, ensure_ascii=False),
        tool_labels_json=json.dumps(tool_labels, ensure_ascii=False),
        tool_tips_json=json.dumps(tool_tips, ensure_ascii=False),
    )
