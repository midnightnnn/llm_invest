from __future__ import annotations

import html
import json
from typing import Any, Callable, Mapping, Sequence

from arena.config import Settings


def _render_recover_button(
    *,
    tenant: str,
    user_email: str | None,
    tenant_settings: Settings,
    is_live_mode: Callable[[Settings | None], bool],
    compact: bool = False,
) -> str:
    recover_tooltip = "슬리브 상태가 꼬였을 때 현재 상태 기준으로 다시 맞춥니다."
    button_cls = (
        "rounded-lg border border-amber-300 bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 "
        "transition hover:bg-amber-100"
        if compact
        else "rounded-xl border border-amber-300 bg-amber-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-amber-600"
    )
    label = "복구" if compact else "Recover Sleeves"
    return (
        f'<form method="post" action="/admin/recover" class="inline-flex items-center gap-1">'
        f'<input type="hidden" name="tenant_id" value="{html.escape(tenant)}" />'
        f'<input type="hidden" name="updated_by" value="{html.escape(user_email or "ui-admin")}" />'
        f'<input type="hidden" name="live" value="{"1" if is_live_mode(tenant_settings) else "0"}" />'
        f'<button type="submit" data-recover-sleeves class="{button_cls}">{label}</button>'
        f'<span class="field-tip">?<span class="tip-body" style="left:auto;right:0;max-width:240px;white-space:normal;">{html.escape(recover_tooltip)}</span></span>'
        "</form>"
    )


def _render_research_status_banner(research_status: Mapping[str, Any]) -> str:
    code = str(research_status.get("code") or "").strip().lower()
    message = str(research_status.get("message") or "").strip()
    if not message:
        return ""
    classes = (
        "mb-4 rounded-2xl border border-emerald-200 bg-emerald-50/80 px-4 py-3 text-sm text-emerald-900"
        if code in {"enabled", "vertex_enabled", "shared_live_tenant"}
        else "mb-4 rounded-2xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-900"
    )
    title = "Research Status"
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
    return (
        f'<div class="{classes}">'
        f'<p class="text-[11px] font-bold uppercase tracking-[0.22em]">{html.escape(title)}</p>'
        f'<p class="mt-1 leading-6">{html.escape(detail)}</p>'
        "</div>"
    )


def _render_agent_card(
    *,
    entry: dict[str, Any],
    api_status: Mapping[str, Any],
    tenant_settings: Settings,
    prompt_text: str,
    risk_fields_meta: Sequence[tuple[str, str, Any]],
    configurable_tools: Sequence[dict[str, Any]],
    provider_key_help: Mapping[str, str],
    provider_api_key_help_html: Callable[[str], str],
) -> str:
    aid = html.escape(str(entry["id"]))
    provider = str(entry.get("provider") or "")
    model = html.escape(str(entry.get("model") or ""))
    has_key = bool(api_status.get(aid) or api_status.get(provider))
    key_badge = (
        '<span class="inline-block rounded-full bg-green-100 px-2 py-0.5 text-[10px] font-bold text-green-700">KEY OK</span>'
        if has_key
        else '<span class="inline-block rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold text-red-600">NO KEY</span>'
    )
    target_market_raw = str(entry.get("target_market") or "").strip().lower() or str(
        tenant_settings.kis_target_market or "us"
    )
    selected_markets = {m.strip() for m in target_market_raw.split(",") if m.strip()}
    market_pills = "".join(
        f'<button type="button" data-market="{m}" class="market-pill inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors select-none'
        f'{" border-blue-400 bg-blue-50 text-blue-700 pill-on" if m in selected_markets else " border-ink-200 bg-white text-ink-500 hover:border-ink-300"}">'
        f"{label}</button>"
        for m, label in (("us", "US"), ("kospi", "KOSPI"))
    )
    sys_prompt = html.escape(str(entry.get("system_prompt") or ""))
    display_prompt = sys_prompt if sys_prompt else html.escape(prompt_text)
    risk_policy = entry.get("risk_policy") or {}
    disabled_tools_set = set(entry.get("disabled_tools") or [])

    risk_inputs = ""
    for risk_key, risk_label, risk_default in risk_fields_meta:
        risk_value = html.escape(str(risk_policy.get(risk_key, "")))
        risk_inputs += (
            f'<label class="text-xs text-ink-600">{risk_label}'
            f'<input data-risk-field="{risk_key}" value="{risk_value}" placeholder="{html.escape(str(risk_default))}"'
            ' class="agent-detail-risk mt-0.5 w-full rounded-lg border border-ink-200 bg-white px-2 py-1 text-xs"/>'
            "</label>"
        )

    tool_toggles = ""
    for tool in configurable_tools:
        tool_id = html.escape(str(tool["tool_id"]))
        tool_category = html.escape(str(tool["category"]))
        tool_tip = html.escape(str(tool.get("description_ko") or tool.get("description") or "")[:60])
        is_on = tool_id not in disabled_tools_set
        on_cls = "bg-blue-500" if is_on else "bg-ink-300"
        knob_cls = "translate-x-4" if is_on else "translate-x-0"
        tool_name = html.escape(str(tool.get("label_ko") or tool.get("tool_id") or tool_id))
        tool_toggles += (
            f'<div class="group relative flex items-center gap-2 rounded-lg border border-ink-200/60 bg-white/70 px-2.5 py-1.5 text-xs cursor-pointer tool-toggle-row" data-tool-row="{tool_id}">'
            f'<button type="button" class="tool-switch relative inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors duration-200 {on_cls}" data-agent-tool-id="{tool_id}" data-on="{str(is_on).lower()}">'
            f'<span class="inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-1 ring-ink-200/30 transition-transform duration-200 mt-0.5 ml-0.5 {knob_cls}"></span></button>'
            f'<span class="font-semibold text-ink-800">{tool_name}</span>'
            f'<span class="text-[10px] text-ink-400">{tool_category}</span>'
            f'<div class="tool-tip-ko invisible group-hover:visible absolute left-0 top-full mt-1 z-50 whitespace-nowrap rounded-lg bg-ink-800 px-2.5 py-1 text-[11px] text-white shadow-lg pointer-events-none">{tool_tip}</div>'
            "</div>"
        )

    return (
        f'<div class="agent-card rounded-xl border border-ink-200/60 bg-white/70 shadow-sm overflow-hidden" data-agent-id="{aid}">'
        '<div class="px-4 py-3 space-y-2.5">'
        '<div class="flex items-center justify-between">'
        f'<div class="flex items-center gap-2"><span class="font-display text-sm font-bold text-ink-900">{aid}</span>{key_badge}</div>'
        '<div class="flex items-center gap-1.5">'
        '<button type="button" class="agent-save-btn hidden rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm">Save</button>'
        '<button type="button" class="agent-toggle-btn text-ink-400 hover:text-ink-700 text-sm px-2 py-1 rounded transition-colors" title="Detail">&#9660;</button>'
        '<button type="button" class="agent-remove-btn text-red-400 hover:text-red-600 text-lg font-bold px-1 transition-colors" title="Remove">&times;</button>'
        "</div></div>"
        f'<input type="hidden" data-field="provider" value="{provider}"/>'
        '<div class="grid gap-x-3 gap-y-1 grid-cols-2 sm:grid-cols-3 items-end">'
        f'<label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Market<span class="field-tip">?<span class="tip-body">에이전트가 거래할 시장. US(나스닥+NYSE) 또는 KOSPI. 복수 선택 시 양쪽 모두 거래합니다.</span></span><div class="mt-0.5 flex gap-1.5 flex-wrap agent-market-pills">{market_pills}</div></label>'
        f'<label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Model<span class="field-tip">?<span class="tip-body">LLM 모델 ID. 예: gpt-5.2, gemini-3-flash-preview, claude-sonnet-4-6</span></span><input type="text" data-field="model" value="{model}" class="mt-0.5 block w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-sm" placeholder="model"/></label>'
        f'<label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">API Key<span class="click-tip">?<span class="tip-body">{provider_key_help.get(provider, provider_api_key_help_html(provider))}</span></span><input type="password" data-field="api_key" placeholder="&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;" class="mt-0.5 block w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-xs" autocomplete="off"/></label>'
        "</div></div>"
        '<div class="agent-detail hidden border-t border-ink-200/40 px-4 py-4 space-y-4 bg-ink-50/30">'
        "<div>"
        '<p class="text-xs font-semibold text-ink-700 mb-1">User Prompt <span class="field-tip">?<span class="tip-body">에이전트의 투자 성향·전략을 지시하는 시스템 프롬프트. 비워두면 기본 프롬프트가 적용됩니다. 에이전트별로 다른 전략을 부여할 수 있습니다.</span></span></p>'
        f'<textarea data-field="system_prompt" rows="10" placeholder="Agent-specific prompt" class="agent-detail-prompt w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 font-mono text-xs resize-y min-h-[60px]">{display_prompt}</textarea>'
        "</div>"
        '<hr class="border-ink-200/30"/>'
        "<div>"
        '<p class="text-xs font-semibold text-ink-700 mb-1">Risk Policy <span class="field-tip">?<span class="tip-body">에이전트별 리스크 한도. 빈칸이면 전역 기본값(placeholder)이 적용됩니다. 최대 주문 금액, 일일 회전율, 종목 비중, 현금 버퍼 등을 개별 조정할 수 있습니다.</span></span></p>'
        f'<div class="grid gap-2 sm:grid-cols-4 agent-risk-grid">{risk_inputs}</div>'
        "</div>"
        "</div>"
        "</div>"
    )


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
    agent_cards_html = "\n".join(
        _render_agent_card(
            entry=entry,
            api_status=api_status,
            tenant_settings=tenant_settings,
            prompt_text=prompt_text,
            risk_fields_meta=risk_fields_meta,
            configurable_tools=configurable_tools,
            provider_key_help=provider_key_help,
            provider_api_key_help_html=provider_api_key_help_html,
        )
        for entry in agents_cfg
    )
    tool_labels = {
        str(tool["tool_id"]): str(tool.get("label_ko") or tool["tool_id"])
        for tool in configurable_tools
    }
    tool_tips = {
        str(tool["tool_id"]): str(tool.get("description_ko") or tool["description"][:60])
        for tool in configurable_tools
    }
    return (
        '<section data-settings-panel="agents" class="settings-panel hidden rounded-[24px] border border-ink-200/60 bg-white/80 p-6 shadow-sm backdrop-blur-md">'
        + '<h3 class="font-display text-lg font-semibold">Agents <span class="info-tip" data-tip="에이전트별 모델·프롬프트·리스크·도구를 독립 설정합니다.&#10;변경 후 Save를 눌러야 반영됩니다.">?</span></h3>'
        + '<div class="mt-4">'
        + _render_research_status_banner(research_status)
        + f'<div id="agents-cards" class="space-y-3">{agent_cards_html}</div>'
        + '<div class="mt-3 flex flex-wrap items-center gap-2">'
        + '<input id="agent-add-name" type="text" placeholder="Agent name" class="rounded-lg border border-ink-200 bg-white px-3 py-1.5 text-sm w-40"/>'
        + '<select id="agent-add-provider" class="rounded-lg border border-ink-200 bg-white px-3 py-1.5 text-sm">'
        + provider_options_html
        + "</select>"
        + '<button type="button" id="agent-add-btn" class="rounded-xl bg-ink-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-ink-600">+ Add Agent</button>'
        + "</div></div>"
        + "<script>(function(){"
        + f"var TOOLS={json.dumps(list(configurable_tools), ensure_ascii=False)};"
        + f"var RISK_FIELDS={json.dumps([(k, l, str(d)) for k, l, d in risk_fields_meta], ensure_ascii=False)};"
        + f"var DEFAULT_MODELS={json.dumps(dict(default_models))};"
        + f"var PROVIDER_KEY_HELP={json.dumps(dict(provider_key_help), ensure_ascii=False)};"
        + f"var DEFAULT_CAP={int(default_capital_krw)};"
        + f'var DEFAULT_MARKET="{html.escape(str(tenant_settings.kis_target_market or "us"))}";'
        + 'var MARKETS=[["us","US (USD)"],["kospi","KOSPI (KRW)"]];'
        + f"var TENANT_ID={json.dumps(tenant)};"
        + f"var DEFAULT_PROMPT={json.dumps(prompt_text, ensure_ascii=False)};"
        + "var RECOVERY_BUTTON_HTML='';"
        + f"var TOOL_LABELS={json.dumps(tool_labels, ensure_ascii=False)};"
        + "function showToast(msg,level){"
        + 'var t=document.createElement("div");'
        + 't.setAttribute("role","status");'
        + 't.style.cssText="position:fixed;right:16px;top:16px;z-index:9999;max-width:420px;padding:12px 14px;border-radius:12px;border:1px solid;box-shadow:0 10px 24px rgba(15,23,42,0.15);backdrop-filter:blur(6px);font-size:13px;font-weight:600;transition:opacity .25s ease,transform .25s ease;opacity:0;transform:translateY(-6px);";'
        + 'if(level==="success"){t.style.background="rgba(236,253,245,0.95)";t.style.borderColor="rgba(16,185,129,0.35)";t.style.color="#065f46";}'
        + 'else{t.style.background="rgba(254,242,242,0.96)";t.style.borderColor="rgba(239,68,68,0.35)";t.style.color="#991b1b";}'
        + "t.textContent=msg;document.body.appendChild(t);"
        + 'requestAnimationFrame(function(){t.style.opacity="1";t.style.transform="translateY(0)";});'
        + 'setTimeout(function(){t.style.opacity="0";t.style.transform="translateY(-6px)";setTimeout(function(){t.remove();},250);},3200);'
        + "}"
        + "function collectCard(card){"
        + "var id=card.dataset.agentId;"
        + 'var provider=card.querySelector("[data-field=provider]").value;'
        + 'var model=card.querySelector("[data-field=model]").value.trim();'
        + 'var apiKey=card.querySelector("[data-field=api_key]").value.trim();'
        + 'var mktArr=[];card.querySelectorAll(".market-pill.pill-on").forEach(function(p){mktArr.push(p.getAttribute("data-market"));});'
        + "var targetMarket=mktArr.join(',');"
        + 'var obj={id:id,provider:provider,model:model,capital_krw:DEFAULT_CAP,api_key:apiKey||"",target_market:targetMarket};'
        + 'var ta=card.querySelector("[data-field=system_prompt]");'
        + "if(ta&&ta.value.trim())obj.system_prompt=ta.value.trim();"
        + "var rp={};"
        + 'card.querySelectorAll("[data-risk-field]").forEach(function(inp){'
        + 'var v=inp.value.trim();if(v)rp[inp.getAttribute("data-risk-field")]=parseFloat(v)||0;'
        + "});"
        + "if(Object.keys(rp).length)obj.risk_policy=rp;"
        + "var off=[];"
        + 'card.querySelectorAll(".tool-switch[data-agent-tool-id]").forEach(function(sw){'
        + 'if(sw.getAttribute("data-on")!=="true")off.push(sw.getAttribute("data-agent-tool-id"));'
        + "});"
        + "obj.disabled_tools=off;"
        + "return obj;"
        + "}"
        + "function bindSave(card){"
        + 'var btn=card.querySelector(".agent-save-btn");'
        + "if(!btn)return;"
        + 'function markDirty(){btn.classList.remove("hidden");btn.textContent="Save";btn.disabled=false;btn.className="agent-save-btn rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm animate-pulse";}'
        + 'card.querySelectorAll("input,select,textarea").forEach(function(el){el.addEventListener("input",markDirty);el.addEventListener("change",markDirty);});'
        + 'btn.addEventListener("click",function(e){'
        + "e.stopPropagation();"
        + "var data=collectCard(card);"
        + 'btn.textContent="Saving...";btn.disabled=true;btn.classList.remove("animate-pulse");btn.classList.add("opacity-60");'
        + 'fetch("/admin/agents/save-one",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({tenant_id:TENANT_ID,agent:data})})'
        + ".then(function(r){return r.json();})"
        + ".then(function(j){"
        + 'if(j.ok){delete card.dataset.agentEphemeral;btn.textContent="Saved!";btn.classList.remove("opacity-60","bg-blue-600");btn.classList.add("bg-green-600");showToast(j.message||"Saved","success");'
        + 'setTimeout(function(){btn.classList.add("hidden");btn.className="agent-save-btn hidden rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm";},1500);}'
        + 'else{btn.textContent="Error";btn.classList.remove("opacity-60");showToast(j.message||"Save failed","error");'
        + 'setTimeout(function(){btn.textContent="Save";btn.disabled=false;},1500);}'
        + "})"
        + '.catch(function(e){btn.textContent="Save";btn.disabled=false;btn.classList.remove("opacity-60");showToast("Network error: "+e.message,"error");});'
        + "});"
        + "}"
        + "function bindToggle(card){"
        + 'var btn=card.querySelector(".agent-toggle-btn");'
        + 'var detail=card.querySelector(".agent-detail");'
        + "if(!btn||!detail)return;"
        + 'btn.addEventListener("click",function(){'
        + "var open=!detail.classList.contains('hidden');"
        + "detail.classList.toggle('hidden');"
        + 'btn.innerHTML=open?"&#9660;":"&#9650;";'
        + "});"
        + "}"
        + "function bindMarketPills(card){"
        + 'card.querySelectorAll(".market-pill").forEach(function(pill){'
        + 'pill.addEventListener("click",function(){'
        + 'var on=pill.classList.toggle("pill-on");'
        + 'pill.className=pill.className.replace(/border-blue-400 bg-blue-50 text-blue-700/g,"").replace(/border-ink-200 bg-white text-ink-500 hover:border-ink-300/g,"");'
        + 'if(on){pill.className+=" border-blue-400 bg-blue-50 text-blue-700";}'
        + 'else{pill.className+=" border-ink-200 bg-white text-ink-500 hover:border-ink-300";}'
        + 'var saveBtn=pill.closest(".agent-card").querySelector(".agent-save-btn");'
        + 'if(saveBtn){saveBtn.classList.remove("hidden");saveBtn.textContent="Save";saveBtn.disabled=false;saveBtn.className="agent-save-btn rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm animate-pulse";}'
        + "});});}"
        + "function bindRemove(card){"
        + 'var btn=card.querySelector(".agent-remove-btn");'
        + "if(!btn)return;"
        + 'btn.addEventListener("click",function(e){'
        + "e.stopPropagation();"
        + "var agentId=card.dataset.agentId||'';"
        + "if(!agentId)return;"
        + 'if(card.dataset.agentEphemeral==="true"){card.remove();return;}'
        + "function requestRemove(forceRemove){"
        + "btn.disabled=true;"
        + 'btn.classList.add("opacity-50","pointer-events-none");'
        + 'fetch("/admin/agents/remove-one",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({tenant_id:TENANT_ID,agent_id:agentId,force:!!forceRemove})})'
        + ".then(function(r){return r.json();})"
        + ".then(function(j){"
        + 'if(j.confirm_required&&!forceRemove){btn.disabled=false;btn.classList.remove("opacity-50","pointer-events-none");if(window.confirm(j.message||"이 에이전트를 제거할까요?")){requestRemove(true);}return;}'
        + 'if(j.ok){card.remove();showToast(j.message||"Agent removed","success");return;}'
        + 'btn.disabled=false;btn.classList.remove("opacity-50","pointer-events-none");showToast(j.message||"Remove failed","error");'
        + "})"
        + '.catch(function(err){btn.disabled=false;btn.classList.remove("opacity-50","pointer-events-none");showToast("Network error: "+err.message,"error");});'
        + "}"
        + "requestRemove(false);"
        + "});"
        + "}"
        + "function toggleSwitch(sw){"
        + 'var on=sw.getAttribute("data-on")==="true";'
        + 'on=!on;sw.setAttribute("data-on",on?"true":"false");'
        + 'var knob=sw.querySelector("span");'
        + 'if(on){sw.className=sw.className.replace(/bg-ink-300/g,"bg-blue-500");knob.className=knob.className.replace(/translate-x-0/g,"translate-x-4");}'
        + 'else{sw.className=sw.className.replace(/bg-blue-500/g,"bg-ink-300");knob.className=knob.className.replace(/translate-x-4/g,"translate-x-0");}'
        + 'var saveBtn=sw.closest(".agent-card").querySelector(".agent-save-btn");'
        + 'if(saveBtn){saveBtn.classList.remove("hidden");saveBtn.textContent="Save";saveBtn.disabled=false;saveBtn.className="agent-save-btn rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm animate-pulse";}'
        + "}"
        + "function bindToolSwitches(card){"
        + 'card.querySelectorAll(".tool-toggle-row").forEach(function(row){'
        + 'row.addEventListener("click",function(e){'
        + 'if(e.target.closest(".tool-tip-ko"))return;'
        + 'var sw=row.querySelector(".tool-switch");if(sw)toggleSwitch(sw);'
        + "});});"
        + 'card.querySelectorAll(".tool-switch").forEach(function(sw){'
        + 'sw.addEventListener("click",function(e){e.stopPropagation();toggleSwitch(sw);});'
        + "});}"
        + "function bindAll(card){bindToggle(card);bindMarketPills(card);bindToolSwitches(card);bindRemove(card);bindSave(card);}"
        + 'document.querySelectorAll(".agent-card").forEach(function(c){bindAll(c);});'
        + 'document.getElementById("agent-add-btn").addEventListener("click",function(){'
        + 'var nameEl=document.getElementById("agent-add-name");'
        + 'var provEl=document.getElementById("agent-add-provider");'
        + 'var v=nameEl.value.trim().toLowerCase()||provEl.value;if(!v)return;nameEl.value="";'
        + "var prov=provEl.value;"
        + 'var riskHtml="";RISK_FIELDS.forEach(function(rf){'
        + "riskHtml+='<label class=\"text-xs text-ink-600\">'+rf[1]+'<input data-risk-field=\"'+rf[0]+'\" value=\"\" placeholder=\"'+rf[2]+'\" class=\"agent-detail-risk mt-0.5 w-full rounded-lg border border-ink-200 bg-white px-2 py-1 text-xs\"\\/><\\/label>';"
        + "});"
        + f"var TOOL_TIPS={json.dumps(tool_tips, ensure_ascii=False)};"
        + 'var toolHtml="";TOOLS.forEach(function(t){'
        + 'var tip=(TOOL_TIPS[t.tool_id]||t.description||"").replace(/"/g,"&quot;").replace(/</g,"&lt;");'
        + 'var toolLabel=(TOOL_LABELS[t.tool_id]||t.tool_id||"").replace(/"/g,"&quot;").replace(/</g,"&lt;");'
        + 'toolHtml+=\'<div class="group relative flex items-center gap-2 rounded-lg border border-ink-200/60 bg-white/70 px-2.5 py-1.5 text-xs cursor-pointer tool-toggle-row" data-tool-row="\'+t.tool_id+\'"><button type="button" class="tool-switch relative inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors duration-200 bg-blue-500" data-agent-tool-id="\'+t.tool_id+\'" data-on="true"><span class="inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-1 ring-ink-200/30 transition-transform duration-200 mt-0.5 ml-0.5 translate-x-4"><\\/span><\\/button><span class="font-semibold text-ink-800">\'+toolLabel+\'<\\/span><span class="text-[10px] text-ink-400">\'+t.category+\'<\\/span><div class="tool-tip-ko invisible group-hover:visible absolute left-0 top-full mt-1 z-50 whitespace-nowrap rounded-lg bg-ink-800 px-2.5 py-1 text-[11px] text-white shadow-lg pointer-events-none">\'+tip+\'<\\/div><\\/div>\';'
        + "});"
        + 'var card=document.createElement("div");'
        + 'card.className="agent-card rounded-xl border border-ink-200/60 bg-white/70 shadow-sm overflow-hidden";'
        + "card.dataset.agentId=v;"
        + 'card.dataset.agentEphemeral="true";'
        + "var defMkts=DEFAULT_MARKET.split(',');"
        + 'var mktPills="";MARKETS.forEach(function(m){'
        + 'var sel=defMkts.indexOf(m[0])>=0;'
        + 'mktPills+=\'<button type="button" data-market="\'+m[0]+\'" class="market-pill inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors select-none\'+(sel?" border-blue-400 bg-blue-50 text-blue-700 pill-on":" border-ink-200 bg-white text-ink-500 hover:border-ink-300")+\'">\'+m[1]+\'<\\/button>\';'
        + "});"
        + 'var promptEsc=DEFAULT_PROMPT.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");'
        + "card.innerHTML="
        + '\'<div class="px-4 py-3 space-y-2.5"><div class="flex items-center justify-between"><div class="flex items-center gap-2"><span class="font-display text-sm font-bold text-ink-900">\'+v+\'</span></div><div class="flex items-center gap-1.5"><button type="button" class="agent-save-btn hidden rounded-lg bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm">Save</button><button type="button" class="agent-toggle-btn text-ink-400 hover:text-ink-700 text-sm px-2 py-1 rounded transition-colors" title="Detail">&#9660;</button><button type="button" class="agent-remove-btn text-red-400 hover:text-red-600 text-lg font-bold px-1 transition-colors" title="Remove">&times;</button></div></div><input type="hidden" data-field="provider" value="\'+prov+\'"/><div class="grid gap-x-3 gap-y-1 grid-cols-2 sm:grid-cols-4 items-end"><label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Market<span class="field-tip">?<span class="tip-body">거래 시장 선택<\\/span><\\/span><div class="mt-0.5 flex gap-1.5 flex-wrap agent-market-pills">\'+mktPills+\'<\\/div><\\/label><label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">Model<span class="field-tip">?<span class="tip-body">LLM 모델 ID<\\/span><\\/span><input type="text" data-field="model" value="\'+(DEFAULT_MODELS[prov]||"")+\'" class="mt-0.5 block w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-sm" placeholder="model"/><\\/label><label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400"><div class="flex items-center justify-between gap-2">Capital &#8361;<span class="inline-flex items-center gap-1"><span class="field-tip">?<span class="tip-body">가상 배정 자본금(원). 변경 후 저장 시 agent별 자본 기준이 다시 반영됩니다.<\\/span><\\/span>\'+RECOVERY_BUTTON_HTML+\'</span></div><input type="number" data-field="capital" value="\'+DEFAULT_CAP+\'" class="mt-0.5 block w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-sm text-right" min="0" step="100000"/><\\/label><label class="text-[10px] font-semibold uppercase tracking-widest text-ink-400">API Key<span class="click-tip">?<span class="tip-body">\'+(PROVIDER_KEY_HELP[prov]||"")+\'<\\/span><\\/span><input type="password" data-field="api_key" placeholder="&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;" class="mt-0.5 block w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 text-xs" autocomplete="off"/><\\/label><\\/div><\\/div><div class="agent-detail hidden border-t border-ink-200/40 px-4 py-4 space-y-4 bg-ink-50/30"><div><p class="text-xs font-semibold text-ink-700 mb-1">User Prompt <span class="field-tip">?<span class="tip-body">에이전트별 투자 전략 프롬프트<\\/span><\\/span><\\/p><textarea data-field="system_prompt" rows="10" placeholder="Agent-specific prompt" class="agent-detail-prompt w-full rounded-lg border border-ink-200 bg-white px-2 py-1.5 font-mono text-xs resize-y min-h-[60px]">\'+promptEsc+\'</textarea></div><hr class="border-ink-200/30"/><div><p class="text-xs font-semibold text-ink-700 mb-1">Risk Policy <span class="field-tip">?<span class="tip-body">에이전트별 리스크 한도 설정<\\/span><\\/span><\\/p><div class="grid gap-2 sm:grid-cols-4 agent-risk-grid">\'+riskHtml+\'<\\/div><\\/div><hr class="border-ink-200/30"/><div><p class="text-xs font-semibold text-ink-700 mb-1">Tools <span class="field-tip">?<span class="tip-body">에이전트 도구 ON/OFF<\\/span><\\/span><\\/p><div class="grid gap-1.5 sm:grid-cols-3 agent-tools-grid max-h-52 overflow-y-auto overflow-x-visible pt-2">\'+toolHtml+\'</div></div></div>\';'
        + 'document.getElementById("agents-cards").appendChild(card);'
        + "bindAll(card);"
        + "});"
        + "})();</script>"
        + "</section>"
    )
