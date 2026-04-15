from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from arena.ui.runtime import _to_bool_token
from arena.ui.templating import render_ui_template


@dataclass(frozen=True)
class CredentialsPanelParts:
    panel_html: str
    kis_section_html: str
    kis_env_options_html: str
    kis_template_real_fields: str
    kis_template_paper_fields: str


def _compact_masked_value(masked: str, *, head: int = 8, tail: int = 6) -> str:
    text = str(masked or "").strip()
    if not text:
        return ""
    if len(text) <= head + tail + 3:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def _build_field_context(name: str, label: str, masked: str = "") -> dict[str, str]:
    masked_str = str(masked or "").strip()
    return {
        "name": name,
        "label": label,
        "masked": masked_str,
        "masked_compact": _compact_masked_value(masked_str),
        "masked_full": masked_str,
    }


def _build_kis_row_context(
    *,
    acct: Mapping[str, Any],
    active_kis_account_no: str,
    allow_real_kis_credentials: bool,
    allow_paper_kis_credentials: bool,
) -> dict[str, Any]:
    env_val = str(acct.get("env") or "real")
    cano_raw = str(acct.get("cano") or "").strip()
    prdt_cd_raw = str(acct.get("prdt_cd") or "01").strip() or "01"
    account_no = f"{cano_raw}-{prdt_cd_raw}" if cano_raw else ""
    account_digits = f"{cano_raw}{prdt_cd_raw}" if cano_raw else ""
    is_active = (
        bool(account_digits)
        and bool(active_kis_account_no)
        and account_digits == active_kis_account_no
    )
    real_selected = allow_real_kis_credentials and env_val != "demo"
    demo_selected = env_val == "demo" or not allow_real_kis_credentials

    real_fields: list[dict[str, str]] = []
    if allow_real_kis_credentials:
        real_fields = [
            _build_field_context("app_key", "APP KEY", str(acct.get("app_key_masked") or "")),
            _build_field_context(
                "app_secret", "APP SECRET", str(acct.get("app_secret_masked") or "")
            ),
        ]
    paper_fields: list[dict[str, str]] = []
    if allow_paper_kis_credentials:
        paper_fields = [
            _build_field_context(
                "paper_app_key", "PAPER APP KEY", str(acct.get("paper_app_key_masked") or "")
            ),
            _build_field_context(
                "paper_app_secret",
                "PAPER APP SECRET",
                str(acct.get("paper_app_secret_masked") or ""),
            ),
        ]

    return {
        "account_no": account_no,
        "account_no_display": account_no or "new account",
        "is_active": is_active,
        "real_selected": real_selected,
        "demo_selected": demo_selected,
        "allow_real_kis_credentials": allow_real_kis_credentials,
        "allow_paper_kis_credentials": allow_paper_kis_credentials,
        "real_fields": real_fields,
        "paper_fields": paper_fields,
    }


def _render_kis_row(
    *,
    acct: Mapping[str, Any],
    active_kis_account_no: str,
    allow_real_kis_credentials: bool,
    allow_paper_kis_credentials: bool,
) -> str:
    row = _build_kis_row_context(
        acct=acct,
        active_kis_account_no=active_kis_account_no,
        allow_real_kis_credentials=allow_real_kis_credentials,
        allow_paper_kis_credentials=allow_paper_kis_credentials,
    )
    return render_ui_template("settings_credentials_kis_row.jinja2", row=row)


def build_credentials_panel(
    *,
    tenant: str,
    credentials_mode_note: str,
    active_kis_account_no: str,
    active_kis_account_no_masked: str,
    kis_meta: Sequence[Mapping[str, Any]],
    allow_real_kis_credentials: bool,
    allow_paper_kis_credentials: bool,
    uses_broker_credentials: bool,
    rows_html: str,
) -> CredentialsPanelParts:
    kis_env_options_html = render_ui_template(
        "settings_credentials_kis_env_options.jinja2",
        allow_real_kis_credentials=allow_real_kis_credentials,
    )
    kis_template_real_fields = render_ui_template(
        "settings_credentials_kis_template_real.jinja2",
        allow_real_kis_credentials=allow_real_kis_credentials,
    )
    kis_template_paper_fields = render_ui_template(
        "settings_credentials_kis_template_paper.jinja2",
        allow_paper_kis_credentials=allow_paper_kis_credentials,
    )

    kis_rows_html = ""
    if uses_broker_credentials:
        kis_rows_html = "".join(
            _render_kis_row(
                acct=acct,
                active_kis_account_no=active_kis_account_no,
                allow_real_kis_credentials=allow_real_kis_credentials,
                allow_paper_kis_credentials=allow_paper_kis_credentials,
            )
            for acct in kis_meta
        ) or _render_kis_row(
            acct={},
            active_kis_account_no=active_kis_account_no,
            allow_real_kis_credentials=allow_real_kis_credentials,
            allow_paper_kis_credentials=allow_paper_kis_credentials,
        )

    kis_section_html = render_ui_template(
        "settings_credentials_kis.jinja2",
        tenant=tenant,
        credentials_mode_note=credentials_mode_note,
        active_kis_account_no_masked=active_kis_account_no_masked,
        uses_broker_credentials=uses_broker_credentials,
        kis_rows_html=kis_rows_html,
    )

    panel_html = render_ui_template("settings_credentials_panel.jinja2")

    return CredentialsPanelParts(
        panel_html=panel_html,
        kis_section_html=kis_section_html,
        kis_env_options_html=kis_env_options_html,
        kis_template_real_fields=kis_template_real_fields,
        kis_template_paper_fields=kis_template_paper_fields,
    )


def _render_mcp_row(server: Mapping[str, Any]) -> str:
    name = html.escape(str(server.get("name") or ""))
    url = html.escape(str(server.get("url") or ""))
    transport = str(server.get("transport") or "sse").strip().lower() or "sse"
    enabled = _to_bool_token(server.get("enabled"), True)
    sse_selected = "selected" if transport == "sse" else ""
    http_selected = "selected" if transport == "streamable_http" else ""
    checked = "checked" if enabled else ""
    return (
        '<div data-mcp-row class="grid gap-2 rounded-xl border border-ink-200/70 bg-white/70 p-3 md:grid-cols-[1fr_1.5fr_150px_110px_auto]">'
        f'<input data-field="name" value="{name}" class="rounded-lg border border-ink-200 bg-white px-2 py-2 text-sm" placeholder="name"/>'
        f'<input data-field="url" value="{url}" class="rounded-lg border border-ink-200 bg-white px-2 py-2 text-sm" placeholder="https://..."/>'
        f'<select data-field="transport" class="rounded-lg border border-ink-200 bg-white px-2 py-2 text-sm"><option value="sse" {sse_selected}>sse</option><option value="streamable_http" {http_selected}>streamable_http</option></select>'
        f'<label class="inline-flex items-center gap-2 rounded-lg border border-ink-200 bg-white px-2 py-2 text-xs text-ink-700"><input data-field="enabled" type="checkbox" {checked} class="h-4 w-4 rounded border-ink-300 text-ink-900"/>enabled</label>'
        + '<button type="button" data-mcp-remove class="rounded-lg border border-red-200 bg-red-50 px-2 py-2 text-xs font-semibold text-red-700 hover:bg-red-100">Remove</button>'
        + "</div>"
    )


def _render_agent_tools_section(
    tenant: str,
    agents_cfg: Sequence[Mapping[str, Any]],
    configurable_tools: Sequence[Mapping[str, Any]],
) -> str:
    """Render per-agent tool toggles with save for the tools management page."""
    import json as _json

    if not configurable_tools or not agents_cfg:
        return ""

    agent_cards_html = ""
    for entry in agents_cfg:
        aid = str(entry.get("agent_id") or entry.get("id") or "").strip().lower()
        if not aid:
            continue
        disabled_raw = entry.get("disabled_tools") or []
        if isinstance(disabled_raw, str):
            try:
                disabled_raw = _json.loads(disabled_raw)
            except Exception:
                disabled_raw = []
        disabled_set = {str(t).strip().lower() for t in disabled_raw}

        toggles = ""
        for tool in configurable_tools:
            tid = html.escape(str(tool["tool_id"]))
            label = html.escape(str(tool.get("label_ko") or tid))
            is_on = tid.lower() not in disabled_set
            on_cls = "bg-blue-500" if is_on else "bg-ink-300"
            knob_cls = "translate-x-4" if is_on else "translate-x-0"
            toggles += (
                f'<div class="flex items-center gap-2 rounded-lg border border-ink-200/60 bg-white/70 px-2.5 py-1.5 text-xs cursor-pointer tool-toggle-row" data-tool-row="{tid}">'
                f'<button type="button" class="tool-switch relative inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors duration-200 {on_cls}" data-agent-tool-id="{tid}" data-on="{str(is_on).lower()}">'
                f'<span class="inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-1 ring-ink-200/30 transition-transform duration-200 mt-0.5 ml-0.5 {knob_cls}"></span></button>'
                f'<span class="font-semibold text-ink-800">{label}</span>'
                "</div>"
            )

        agent_cards_html += (
            f'<div class="rounded-xl border border-ink-200/50 bg-white/70 p-3 tools-agent-card" data-tools-agent="{html.escape(aid)}">'
            f'<div class="flex items-center justify-between mb-2">'
            f'<p class="text-xs font-bold text-ink-800">{html.escape(aid.upper())}</p>'
            f'<button type="button" class="tools-agent-save hidden rounded-lg bg-blue-600 px-3 py-1 text-[10px] font-semibold text-white hover:bg-blue-700 transition-colors shadow-sm">Save</button>'
            f'</div>'
            f'<div class="grid gap-1.5 sm:grid-cols-3 max-h-52 overflow-y-auto pt-1">{toggles}</div>'
            f'</div>'
        )

    script = (
        "<script>(function(){"
        f"var TENANT={_json.dumps(tenant)};"
        "document.querySelectorAll('.tools-agent-card').forEach(function(card){"
        "var agentId=card.dataset.toolsAgent;"
        "var saveBtn=card.querySelector('.tools-agent-save');"
        "card.querySelectorAll('.tool-switch').forEach(function(sw){"
        "sw.addEventListener('click',function(){"
        "var on=sw.dataset.on==='true';"
        "sw.dataset.on=on?'false':'true';"
        "sw.classList.toggle('bg-blue-500',!on);"
        "sw.classList.toggle('bg-ink-300',on);"
        "var knob=sw.querySelector('span');"
        "if(knob){knob.classList.toggle('translate-x-4',!on);knob.classList.toggle('translate-x-0',on);}"
        "if(saveBtn)saveBtn.classList.remove('hidden');"
        "});"
        "});"
        "card.querySelectorAll('.tool-toggle-row').forEach(function(row){"
        "row.addEventListener('click',function(e){"
        "if(e.target.closest('.tool-switch'))return;"
        "var sw=row.querySelector('.tool-switch');"
        "if(sw)sw.click();"
        "});"
        "});"
        "if(saveBtn)saveBtn.addEventListener('click',function(){"
        "var disabled=[];"
        "card.querySelectorAll('.tool-switch').forEach(function(sw){"
        "if(sw.dataset.on!=='true')disabled.push(sw.dataset.agentToolId);"
        "});"
        "saveBtn.textContent='Saving...';"
        "saveBtn.disabled=true;"
        "fetch('/admin/agents/save-one',{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({tenant_id:TENANT,agent:{id:agentId,disabled_tools:disabled}})})"
        ".then(function(r){return r.json();})"
        ".then(function(d){"
        "saveBtn.textContent='Save';"
        "saveBtn.disabled=false;"
        "if(d&&d.ok){saveBtn.classList.add('hidden');return;}"
        "if(d&&d.message){window.alert(d.message);}"
        "})"
        ".catch(function(){"
        "saveBtn.textContent='Save';"
        "saveBtn.disabled=false;"
        "});"
        "});"
        "});"
        "})();</script>"
    )

    return (
        '<div class="mt-6 rounded-[20px] border border-ink-200/50 bg-white/80 p-5">'
        '<h4 class="font-display text-sm font-semibold text-ink-900">'
        '에이전트별 도구 On/Off'
        '</h4>'
        f'<div class="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{agent_cards_html}</div>'
        '</div>'
        + script
    )


def _build_tool_catalog_script(
    tools: Sequence[Mapping[str, Any]],
    tenant: str = "",
    agents_cfg: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Build the JS for the interactive tool catalog with 3-tab detail panel."""
    import json as _json

    cat_order = {"context": 0, "quant": 1, "macro": 2, "sentiment": 3}
    cat_labels = {"context": "Context", "quant": "Quant", "macro": "Macro", "sentiment": "Sentiment"}

    # Group tools by category
    by_cat: dict[str, list[Mapping[str, Any]]] = {}
    for t in tools:
        cat = str(t.get("category") or "other")
        by_cat.setdefault(cat, []).append(t)
    sorted_cats = sorted(by_cat.keys(), key=lambda c: cat_order.get(c, 99))

    # Build sidebar tool list HTML
    sidebar_items = ""
    first_id = ""
    for cat in sorted_cats:
        label = cat_labels.get(cat, cat.title())
        sidebar_items += f'<p class="hidden md:block mt-3 first:mt-0 mb-1 px-2 text-[9px] font-bold uppercase tracking-[0.25em] text-ink-400/60">{html.escape(label)}</p>'
        for t in by_cat[cat]:
            tid = html.escape(str(t["tool_id"]))
            lbl = html.escape(str(t.get("label_ko") or tid))
            tier = str(t.get("tier") or "")
            tier_dot = '<span class="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0"></span>' if tier == "core" else '<span class="w-1.5 h-1.5 rounded-full bg-ink-300 shrink-0"></span>'
            if not first_id:
                first_id = tid
            sidebar_items += (
                f'<button type="button" data-tool-select="{tid}" '
                f'class="tool-catalog-item flex items-center gap-2 shrink-0 md:shrink md:w-full text-left rounded-lg px-2.5 py-2 text-[12px] font-medium text-ink-600 '
                f'transition-all duration-150 hover:bg-ink-50 hover:text-ink-900">'
                f'{tier_dot}<span class="truncate">{lbl}</span></button>'
            )

    # Build per-agent disabled_tools map
    import json as _json
    agents_disabled: dict[str, list[str]] = {}
    agent_ids_list: list[str] = []
    for entry in (agents_cfg or []):
        aid = str(entry.get("agent_id") or entry.get("id") or "").strip().lower()
        if not aid:
            continue
        agent_ids_list.append(aid)
        disabled_raw = entry.get("disabled_tools") or []
        if isinstance(disabled_raw, str):
            try:
                disabled_raw = _json.loads(disabled_raw)
            except Exception:
                disabled_raw = []
        agents_disabled[aid] = [str(t).strip().lower() for t in disabled_raw]

    # Serialize tool data for JS
    tools_json = _json.dumps(
        {str(t["tool_id"]): {
            "tool_id": t["tool_id"],
            "label_ko": t.get("label_ko") or t["tool_id"],
            "description": t.get("description") or "",
            "description_ko": t.get("description_ko") or "",
            "category": t.get("category") or "",
            "tier": t.get("tier") or "",
            "params": t.get("params") or [],
            "source": t.get("source") or "",
        } for t in tools},
        ensure_ascii=False,
    ).replace("<", "\\u003c").replace("</", "\\u003c/")
    agents_json = _json.dumps(agents_disabled, ensure_ascii=False)
    agent_ids_json = _json.dumps(agent_ids_list, ensure_ascii=False)
    tenant_json = _json.dumps(tenant)

    # Inline style for visible scrollbar (styled-scrollbar is too thin)
    _scroll_css = (
        '<style>'
        '#toolCatalogGrid .tool-scroll::-webkit-scrollbar{width:6px;height:6px;}'
        '#toolCatalogGrid .tool-scroll::-webkit-scrollbar-track{background:transparent;}'
        '#toolCatalogGrid .tool-scroll::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px;}'
        '#toolCatalogGrid .tool-scroll::-webkit-scrollbar-thumb:hover{background:#94a3b8;}'
        '#toolCatalogGrid .tool-scroll{scrollbar-width:thin;scrollbar-color:#cbd5e1 transparent;}'
        '</style>'
    )

    return (
        _scroll_css
        # Container
        + '<div class="mt-6 rounded-[20px] border border-ink-200/40 '
        'bg-[linear-gradient(135deg,rgba(255,255,255,0.97),rgba(248,250,252,0.95))] '
        'shadow-[0_8px_32px_rgba(15,23,42,0.04)] overflow-hidden">'
        '<div class="flex items-center gap-2 border-b border-ink-200/30 px-5 py-3">'
        '<svg class="w-4 h-4 text-ink-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M11.42 15.17l-5.66-5.66a8 8 0 1111.31 0l-5.65 5.66z"/></svg>'
        '<h4 class="text-sm font-bold text-ink-900 tracking-tight">도구 카탈로그</h4>'
        f'<span class="ml-auto text-[10px] text-ink-400 font-mono">{len(tools)} tools</span>'
        '</div>'
        '<div class="grid grid-cols-1 md:grid-cols-[200px_1fr] md:h-[540px]" id="toolCatalogGrid">'
        # Sidebar — horizontal scroll on mobile, vertical on desktop
        '<div class="tool-scroll border-b md:border-b-0 md:border-r border-ink-200/30 py-2 px-1.5 overflow-x-auto md:overflow-x-hidden overflow-y-hidden md:overflow-y-auto bg-white/50 flex md:block gap-1 md:gap-0 whitespace-nowrap md:whitespace-normal">'
        + sidebar_items
        + '</div>'
        # Detail panel — scrollable body
        '<div class="flex flex-col min-h-0 h-[400px] md:h-auto" id="toolDetailPanel">'
        '<div id="toolDetailHeader" class="border-b border-ink-200/30 px-4 md:px-5 py-3 shrink-0"></div>'
        '<div class="flex border-b border-ink-200/20 shrink-0">'
        '<button type="button" data-tool-tab="desc" class="tool-detail-tab px-3 md:px-4 py-2 text-[11px] font-semibold text-ink-500 border-b-2 border-transparent transition-all hover:text-ink-700">설명</button>'
        '<button type="button" data-tool-tab="schema" class="tool-detail-tab px-3 md:px-4 py-2 text-[11px] font-semibold text-ink-500 border-b-2 border-transparent transition-all hover:text-ink-700">요청 · 응답</button>'
        '<button type="button" data-tool-tab="code" class="tool-detail-tab px-3 md:px-4 py-2 text-[11px] font-semibold text-ink-500 border-b-2 border-transparent transition-all hover:text-ink-700">코드</button>'
        '</div>'
        '<div id="toolDetailBody" class="tool-scroll flex-grow overflow-y-auto p-4 md:p-5 min-h-0"></div>'
        '</div>'
        '</div></div>'
        # Script
        + "<script>(function(){"
        + f"var TOOLS={tools_json};"
        + f"var AGENTS_DISABLED={agents_json};"
        + f"var AGENT_IDS={agent_ids_json};"
        + f"var TENANT={tenant_json};"
        + r"""
var activeToolId="";
var activeTab="desc";
function esc(s){return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

function isToolEnabled(agentId,toolId){
  var d=AGENTS_DISABLED[agentId]||[];
  return d.indexOf(toolId.toLowerCase())<0;
}
function toggleAgent(agentId,toolId,btnEl){
  var d=AGENTS_DISABLED[agentId]||[];
  var tidL=toolId.toLowerCase();
  var idx=d.indexOf(tidL);
  if(idx>=0){d.splice(idx,1);}else{d.push(tidL);}
  AGENTS_DISABLED[agentId]=d;
  if(btnEl){btnEl.style.opacity="0.5";btnEl.style.pointerEvents="none";}
  fetch('/admin/agents/save-one',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({tenant_id:TENANT,agent:{id:agentId,disabled_tools:d}})
  }).then(function(r){return r.json();}).then(function(resp){
    if(btnEl){btnEl.style.opacity="";btnEl.style.pointerEvents="";}
    if(resp&&!resp.ok&&resp.message){_toast(resp.message,"error");}
    else{_toast(agentId.toUpperCase()+" — "+(isToolEnabled(agentId,toolId)?"ON":"OFF"),"success");}
  }).catch(function(){
    if(btnEl){btnEl.style.opacity="";btnEl.style.pointerEvents="";}
    _toast("저장 실패","error");
  });
}
function _toast(msg,level){
  var t=document.createElement("div");
  t.style.cssText="position:fixed;right:16px;top:16px;z-index:9999;max-width:300px;padding:10px 14px;border-radius:12px;font-size:12px;font-weight:600;opacity:0;transform:translateY(-6px);transition:opacity .2s,transform .2s;";
  if(level==="error"){t.style.background="rgba(254,242,242,0.95)";t.style.color="#991b1b";t.style.border="1px solid rgba(239,68,68,0.3)";}
  else{t.style.background="rgba(236,253,245,0.95)";t.style.color="#065f46";t.style.border="1px solid rgba(16,185,129,0.3)";}
  t.textContent=msg;document.body.appendChild(t);
  requestAnimationFrame(function(){t.style.opacity="1";t.style.transform="translateY(0)";});
  setTimeout(function(){t.style.opacity="0";t.style.transform="translateY(-6px)";setTimeout(function(){t.remove();},200);},2000);
}

function selectTool(tid){
  activeToolId=tid;
  document.querySelectorAll(".tool-catalog-item").forEach(function(el){
    var isSel=el.dataset.toolSelect===tid;
    el.classList.toggle("bg-ink-900",isSel);
    el.classList.toggle("!text-white",isSel);
    el.classList.toggle("shadow-sm",isSel);
    el.classList.toggle("hover:bg-ink-50",!isSel);
  });
  var t=TOOLS[tid];if(!t)return;
  var hdr=document.getElementById("toolDetailHeader");
  var tierBadge=t.tier==="core"
    ?'<span class="rounded-full bg-blue-100 px-2 py-0.5 text-[9px] font-bold text-blue-700 uppercase tracking-wider">Core</span>'
    :'<span class="rounded-full bg-ink-100 px-2 py-0.5 text-[9px] font-bold text-ink-500 uppercase tracking-wider">Optional</span>';
  var catBadge='<span class="rounded-full border border-ink-200/50 px-2 py-0.5 text-[9px] font-medium text-ink-500 uppercase tracking-wider">'+esc(t.category)+'</span>';
  // Agent toggles
  var toggles="";
  AGENT_IDS.forEach(function(aid){
    var on=isToolEnabled(aid,tid);
    var onCls=on?"bg-emerald-500":"bg-ink-300";
    var knobCls=on?"translate-x-3.5":"translate-x-0";
    toggles+=
      '<button type="button" class="agent-tool-toggle inline-flex items-center gap-1.5 rounded-lg border border-ink-200/40 bg-white/70 px-2 py-1 text-[10px] transition-all hover:shadow-sm" data-agent-id="'+esc(aid)+'" data-tool-id="'+esc(tid)+'">' +
      '<span class="agent-toggle-track relative inline-flex h-4 w-8 rounded-full transition-colors duration-200 '+onCls+'">' +
      '<span class="agent-toggle-knob inline-block h-3 w-3 transform rounded-full bg-white shadow-sm ring-1 ring-ink-200/20 transition-transform duration-200 mt-0.5 ml-0.5 '+knobCls+'"></span></span>' +
      '<span class="font-semibold text-ink-700">'+esc(aid.toUpperCase())+'</span></button>';
  });
  var toggleRow=AGENT_IDS.length?'<div class="flex flex-wrap items-center gap-1.5 mt-2" id="agentTogglesRow">'+toggles+'</div>':"";
  hdr.innerHTML='<div class="flex items-center gap-2 flex-wrap"><h5 class="text-base font-bold text-ink-900 tracking-tight">'+esc(t.label_ko)+'</h5>'+tierBadge+catBadge+'</div><p class="mt-1 text-[11px] text-ink-400 font-mono">'+esc(t.tool_id)+'</p>'+toggleRow;
  // Bind toggle clicks
  document.querySelectorAll(".agent-tool-toggle").forEach(function(btn){
    btn.addEventListener("click",function(){
      var aid=btn.dataset.agentId;
      var toolIdVal=btn.dataset.toolId;
      toggleAgent(aid,toolIdVal,btn);
      var nowOn=isToolEnabled(aid,toolIdVal);
      var track=btn.querySelector(".agent-toggle-track");
      var knob=btn.querySelector(".agent-toggle-knob");
      if(track){track.classList.toggle("bg-emerald-500",nowOn);track.classList.toggle("bg-ink-300",!nowOn);}
      if(knob){knob.classList.toggle("translate-x-3.5",nowOn);knob.classList.toggle("translate-x-0",!nowOn);}
    });
  });
  showTab(activeTab);
}

var RESPONSE_EXAMPLES={
  search_past_experiences:'[\n  {"event_id":"evt_123","summary":"Strong tech rally, bought QQQ","score":0.85,"memory_date":"2024-01-15"}\n]',
  search_peer_lessons:'[\n  {"event_id":"evt_456","agent_id":"gpt","summary":"Avoid energy sector in bear markets","author_id":"gpt","memory_source":"memory_compaction"}\n]',
  get_research_briefing:'[\n  {"title":"Tech Sector Outlook","content":"AI momentum continues...","category":"sector","source":"research_team"}\n]',
  save_memory:'{"status":"saved","event_type":"manual_note","summary":"Tesla momentum breaking down"}',
  portfolio_diagnosis:'{\n  "risk_contribution":[{"ticker":"AAPL","rc":0.35},{"ticker":"MSFT","rc":0.28}],\n  "concentration_top3":0.65, "hhi":0.18,\n  "mdd":{"days":60,"value":-0.125},\n  "hrp_allocation":{"status":"ready","strategy":"hrp",\n    "hrp_weights":[{"ticker":"AAPL","hrp_weight":0.35}]}\n}',
  screen_market:'[\n  {"ticker":"AAPL","bucket":"momentum","bucket_rank":1,"score":2.14,\n   "ret_20d":0.0823,"volatility_20d":0.18,"reason":"Strong multi-window momentum"},\n  {"ticker":"PBR","bucket":"value","bucket_rank":2,"score":1.34,\n   "per":6.83,"pbr":1.76,"reason":"Valuation support: PER 6.83, PBR 1.76"}\n]',
  optimize_portfolio:'{\n  "tickers":["AAPL","MSFT","GOOGL"], "strategy":"sharpe",\n  "weights":{"AAPL":0.35,"MSFT":0.40,"GOOGL":0.25},\n  "expected_return_daily":0.0008, "sharpe_daily":0.065,\n  "backtest_mdd":{"days":60,"value":-0.082}\n}',
  forecast_returns:'[\n  {"ticker":"MSFT","prob_up":0.72,"model_votes_up":5,"model_votes_total":7,\n   "consensus":"bullish","exp_return_period":0.045,"forecast_model":"ensemble_wmae"},\n  {"ticker":"PBR","prob_up":0.64,"model_votes_up":4,"model_votes_total":7,\n   "consensus":"bullish","exp_return_period":0.031,"best_base_model":"nhits"}\n]',
  technical_signals:'{\n  "ticker":"TSLA","price":245.67,\n  "rsi_14":68.5,"rsi_state":"overbought",\n  "macd":{"line":0.0034,"signal":0.0028,"state":"bullish"},\n  "trend_state":"uptrend"\n}',
  sector_summary:'[\n  {"sector":"Technology","avg_ret":0.0512,"avg_vol":0.195,\n   "tickers":["AAPL","MSFT","NVDA"]}\n]',
  get_fundamentals:'{\n  "rows":[{"ticker":"AAPL","market":"us","per":28.5,"pbr":45.2,"eps":6.15,"bps":3.82}],\n  "errors":[]\n}',
  index_snapshot:'{\n  "indices":[\n    {"symbol":"SPX","name":"S&P 500","close":5234.8,"type":"index"},\n    {"symbol":"US10Y","value":4.28,"unit":"%","type":"bond_yield"}\n  ],\n  "source":"kis_index+fred"\n}',
  fear_greed_index:'{\n  "fear_greed_score":72.5, "regime":"Greed",\n  "regime_label":"risk_on",\n  "sub_components":{\n    "volatility":{"score":72.5,"weight":0.35},\n    "breadth":{"score":65.0,"weight":0.25}\n  }\n}',
  earnings_calendar:'{\n  "ticker":"AAPL","count":3,\n  "rows":[{"date":"2024-04-25","symbol":"AAPL","event_type":"earnings",\n    "eps_forecast":"1.53","time":"After Hours"}],\n  "source":"nasdaq_calendar_api"\n}',
  fetch_reddit_sentiment:'[\n  {"title":"AAPL breaking out to new highs","score":2847,\n   "num_comments":512,"subreddit":"wallstreetbets"}\n]',
  fetch_sec_filings:'[\n  {"form_type":"10-K","filed_date":"2024-02-28","entity":"Apple Inc",\n   "description":"Annual Report"}\n]',
  macro_snapshot:'{\n  "indicators":{\n    "fed_funds_rate":{"value":5.33,"unit":"%"},\n    "cpi_yoy":{"value":3.24,"unit":"%"},\n    "unemployment_rate":{"value":3.9,"unit":"%"},\n    "treasury_10y":{"value":4.28,"unit":"%"}\n  },\n  "source":"fred"\n}'
};

function showTab(tab){
  activeTab=tab;
  document.querySelectorAll(".tool-detail-tab").forEach(function(el){
    var isSel=el.dataset.toolTab===tab;
    el.classList.toggle("border-ink-900",isSel);
    el.classList.toggle("text-ink-900",isSel);
    el.classList.toggle("border-transparent",!isSel);
  });
  var body=document.getElementById("toolDetailBody");
  body.scrollTop=0;
  var t=TOOLS[activeToolId];
  if(!t){body.innerHTML='<p class="text-sm text-ink-400 text-center py-8">도구를 선택하세요</p>';return;}
  if(tab==="desc"){
    var descHtml=esc(t.description_ko).replace(/\. /g,'.<br/>');
    body.innerHTML=
      '<div class="text-[13px] text-ink-700 leading-[1.85]">'+descHtml+'</div>';
  } else if(tab==="schema"){
    var params=t.params||[];
    if(!params.length){
      body.innerHTML='<div class="text-center py-8"><p class="text-sm text-ink-400">파라미터 정보를 가져올 수 없습니다.</p></div>';
      return;
    }
    var rows="";
    params.forEach(function(p){
      var reqBadge=p.required
        ?'<span class="rounded bg-rose-100 px-1.5 py-0.5 text-[9px] font-bold text-rose-600">required</span>'
        :'<span class="rounded bg-ink-100 px-1.5 py-0.5 text-[9px] font-medium text-ink-400">optional</span>';
      var typeStr=p.type?esc(p.type):'<span class="text-ink-300">any</span>';
      var defStr=p.default?'<span class="font-mono text-ink-500">= '+esc(p.default)+'</span>':'';
      rows+='<tr class="border-b border-ink-100/40 hover:bg-ink-50/50 transition-colors">'+
        '<td class="px-3 py-2.5 text-[12px] font-semibold font-mono text-ink-800">'+esc(p.name)+'</td>'+
        '<td class="px-3 py-2.5 text-[11px] text-ink-500 font-mono">'+typeStr+'</td>'+
        '<td class="px-3 py-2.5 text-[11px]">'+reqBadge+" "+defStr+'</td></tr>';
    });
    var exampleJson=RESPONSE_EXAMPLES[t.tool_id]||'{"result": "..."}';
    body.innerHTML=
      '<div class="space-y-5">'+
      '<div><p class="text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-400 mb-2">Request Parameters</p>'+
      '<div class="rounded-xl border border-ink-200/40 overflow-hidden">'+
      '<table class="min-w-full text-left"><thead><tr class="bg-ink-50/60 text-[9px] font-bold uppercase tracking-widest text-ink-400">'+
      '<th class="px-3 py-2">Name</th><th class="px-3 py-2">Type</th><th class="px-3 py-2">Info</th></tr></thead>'+
      '<tbody>'+rows+'</tbody></table></div></div>'+
      '<div><p class="text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-400 mb-2">Response Example</p>'+
      '<div class="rounded-xl bg-[#0f172a] border border-ink-700/30 overflow-hidden">'+
      '<pre class="p-4 text-[11px] leading-[1.6] text-emerald-300 font-mono overflow-x-auto styled-scrollbar" style="tab-size:2;">'+esc(exampleJson)+'</pre></div></div>'+
      '</div>';
  } else if(tab==="code"){
    var src=t.source;
    if(!src){
      body.innerHTML='<div class="text-center py-8"><p class="text-sm text-ink-400">소스 코드를 가져올 수 없습니다.</p></div>';
      return;
    }
    body.innerHTML=
      '<div class="rounded-xl bg-[#0f172a] border border-ink-700/30 overflow-hidden shadow-inner">'+
      '<div class="flex items-center gap-2 px-4 py-2 border-b border-ink-700/30">'+
      '<span class="w-2 h-2 rounded-full bg-red-400/60"></span>'+
      '<span class="w-2 h-2 rounded-full bg-yellow-400/60"></span>'+
      '<span class="w-2 h-2 rounded-full bg-green-400/60"></span>'+
      '<span class="ml-2 text-[10px] text-ink-400 font-mono">'+esc(t.tool_id)+'.py</span></div>'+
      '<pre class="p-4 text-[11.5px] leading-[1.65] text-slate-300 font-mono overflow-x-auto overflow-y-auto styled-scrollbar" style="tab-size:4;">'+esc(src)+'</pre></div>';
  }
}

document.querySelectorAll(".tool-catalog-item").forEach(function(el){
  el.addEventListener("click",function(){selectTool(el.dataset.toolSelect);});
});
document.querySelectorAll(".tool-detail-tab").forEach(function(el){
  el.addEventListener("click",function(){showTab(el.dataset.toolTab);});
});
"""
        + f'if(TOOLS["{first_id}"])selectTool("{first_id}");'
        + "})();</script>"
    )


def build_mcp_panel(
    *,
    tenant: str,
    mcp_servers: Sequence[Mapping[str, Any]],
    agents_cfg: Sequence[Mapping[str, Any]] | None = None,
    configurable_tools: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    mcp_rows = "".join(_render_mcp_row(server) for server in mcp_servers)
    tool_catalog = _build_tool_catalog_script(configurable_tools or [], tenant=tenant, agents_cfg=agents_cfg)
    return (
        '<section data-settings-panel="mcp" class="settings-panel hidden space-y-5">'
        # Tool catalog with integrated agent toggles
        + tool_catalog
        # MCP servers (collapsible)
        + '<details class="rounded-[20px] border border-ink-200/40 bg-white/80 shadow-sm backdrop-blur-md">'
        + '<summary class="cursor-pointer px-5 py-3 text-sm font-bold text-ink-900 hover:bg-ink-50/50 transition-colors rounded-[20px]">'
        + 'MCP Servers <span class="text-[10px] font-normal text-ink-400 ml-2">외부 MCP 서버 연결</span></summary>'
        + '<div class="px-5 pb-5">'
        + '<details class="mb-4 rounded-xl bg-blue-50/80 border border-blue-200/40 text-[12px] text-blue-900">'
        + '<summary class="cursor-pointer px-4 py-2.5 font-semibold hover:bg-blue-100/50 transition-colors rounded-xl select-none">MCP 서버 사용 가이드</summary>'
        + '<div class="px-4 pb-3 leading-relaxed space-y-2">'
        + '<p>외부 MCP 서버를 연결하면 에이전트가 기본 내장 도구 외에 <b>추가 도구</b>를 사용할 수 있습니다. '
        + '예: 사내 DB 조회, Slack 알림, 커스텀 분석 API 등.</p>'
        + '<p class="font-semibold">연결 방법</p>'
        + '<ol class="list-decimal ml-4 space-y-1">'
        + '<li>MCP 서버를 배포하고 SSE 또는 Streamable HTTP 엔드포인트를 확보합니다.</li>'
        + '<li>아래 <b>+ Add</b>를 눌러 이름과 URL을 입력합니다.</li>'
        + '<li><b>Save</b>하면 다음 에이전트 사이클부터 해당 도구가 자동으로 등록됩니다.</li>'
        + '</ol>'
        + '<p class="text-[11px] text-blue-700/70 mt-1">Transport: <code class="bg-blue-100 px-1 rounded">sse</code>(기본) 또는 <code class="bg-blue-100 px-1 rounded">streamable_http</code>. enabled를 끄면 연결 해제 없이 비활성화됩니다.</p>'
        + '</div></details>'
        + '<form id="mcp-servers-form" class="grid gap-3" method="post" action="/admin/tools/mcp">'
        + f'<input type="hidden" name="tenant_id" value="{html.escape(tenant)}"/>'
        + '<input type="hidden" id="mcp_servers_json_field" name="mcp_servers_json" value="[]"/>'
        + '<div data-mcp-rows class="grid gap-2">'
        + mcp_rows
        + "</div>"
        + '<div class="flex flex-wrap items-center gap-2">'
        + '<button type="button" data-mcp-add class="rounded-xl border border-brand-200 bg-brand-50 px-3 py-2 text-sm font-semibold text-brand-800 hover:bg-brand-100">+ Add</button>'
        + '<button class="rounded-xl bg-ink-900 px-4 py-2 text-sm font-medium text-white hover:bg-ink-700" type="submit">Save</button>'
        + "</div>"
        + "</form>"
        + "</div></details>"
        + "</section>"
    )
