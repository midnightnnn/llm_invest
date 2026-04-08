from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.config import research_generation_status
from arena.ui.templating import render_ui_template
from arena.ui.routes.viewer import ViewerRouteDeps


def register_board_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/board", response_class=HTMLResponse)
    def board(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent_id filter"),
        date: str = Query(default="", description="date filter YYYY-MM-DD"),
        limit: int = Query(default=20, ge=1, le=400),
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/board?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        offset = (page - 1) * limit
        agent_key = ",".join(sorted(scoped_agent_ids))
        date_filter = date.strip()[:10] if date.strip() else datetime.now(deps.kst).strftime("%Y-%m-%d")
        fut_board_rows = deps.executor.submit(
            deps.cached_fetch,
            f"board:{tenant}:{agent_key}:{limit}:{offset}:{token or ''}:{date_filter or ''}",
            deps.fetch_board,
            tenant_id=tenant,
            limit=limit,
            offset=offset,
            agent_id=token,
            agent_ids=scoped_agent_ids,
            start_date=date_filter,
            end_date=date_filter,
        )
        fut_board_registry = deps.executor.submit(deps.get_default_registry, tenant)
        rows = fut_board_rows.result()
        has_next = len(rows) == limit
        selected_agent_id = token or ""
        research_status = research_generation_status(deps.settings_for_tenant(tenant))
        empty_state_message = "게시글이 없습니다."
        if not rows:
            code = str(research_status.get("code") or "").strip().lower()
            if code == "missing_gemini_key":
                empty_state_message = (
                    "아직 게시글이 없습니다. 이 테넌트는 Gemini 키가 없어 "
                    "새로운 리서치 브리핑 생성도 비활성화되어 있습니다."
                )
            elif code == "disabled_by_config":
                empty_state_message = (
                    "아직 게시글이 없습니다. 이 테넌트는 설정상 리서치 브리핑 생성을 꺼둔 상태입니다."
                )

        def _post_ts_iso(row: dict[str, object]) -> str:
            value = row.get("created_at")
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.isoformat()
            return str(value or "")

        posts = [
            {
                "agent_id": str(row.get("agent_id") or ""),
                "ts_iso": _post_ts_iso(row),
                "cycle_id": str(row.get("cycle_id") or ""),
                "created_at_label": deps.fmt_ts(row.get("created_at")),
                "title": str(row.get("title") or ""),
                "body_html": deps.md_block(row.get("body"), classes="mt-3 text-sm leading-relaxed text-ink-700"),
            }
            for row in rows
        ]

        prev_page = page - 1 if page > 1 else 1
        next_page = page + 1 if has_next else page

        page_params: dict[str, str | int] = {"page": prev_page if page <= 1 else prev_page}
        if date_filter:
            page_params["date"] = date_filter
        if deps.auth_enabled:
            page_params["tenant_id"] = tenant
        if selected_agent_id:
            page_params["agent_id"] = selected_agent_id
        prev_url = "/board?" + urlencode(page_params | {"page": prev_page})
        next_url = "/board?" + urlencode(page_params | {"page": next_page})
        try:
            board_tool_categories = {
                entry.tool_id: entry.category
                for entry in fut_board_registry.result().list_entries(include_disabled=True)
            }
        except Exception:
            board_tool_categories = {}

        tool_accordion_js = (
            '<script>'
            + f'var _boardTenant={json.dumps(tenant)};'
            + f'var CAT_MAP={json.dumps(board_tool_categories, ensure_ascii=False)};'
            + r"""
var CAT_COLORS={
  quant:"bg-blue-100 text-blue-700",
  macro:"bg-sky-100 text-sky-700",
  sentiment:"bg-green-100 text-green-700",
  performance:"bg-purple-100 text-purple-700",
  perf:"bg-purple-100 text-purple-700",
  context:"bg-amber-100 text-amber-700",
  other:"bg-gray-100 text-gray-600"
};
function escapeHtml(value){
  return String(value==null?"":value)
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;")
    .replace(/'/g,"&#39;");
}
function fmtMs(ms){return ms>=1000?(ms/1000).toFixed(1)+"s":Math.round(ms)+"ms";}
function summarizeArgs(args){
  if(!args||typeof args!=="object")return"";
  var parts=[];
  Object.keys(args).slice(0,3).forEach(function(k){
    var v=args[k];
    if(Array.isArray(v))v=v.slice(0,4).join(", ")+(v.length>4?" ...":"");
    else if(typeof v==="object"&&v!==null)v=JSON.stringify(v).slice(0,60);
    else v=String(v).slice(0,60);
    parts.push(k+": "+v);
  });
  return parts.join(" | ");
}
function groupParallel(evts){
  var groups=[],current=null;
  evts.forEach(function(ev){
    if(!ev.started_at){groups.push([ev]);return;}
    var end=ev.started_at+(ev.elapsed_ms||0);
    if(!current||ev.started_at>=current.end){
      current={start:ev.started_at,end:end,items:[ev]};
      groups.push(current.items);
    }else{
      current.items.push(ev);
      current.end=Math.max(current.end,end);
    }
  });
  return groups;
}
function summarizeResult(r){
  if(!r)return"";
  if(typeof r==="object"&&r!==null){
    if(Array.isArray(r.keys))return"keys: "+r.keys.join(", ");
    if(r.len!=null)return"rows: "+r.len+(r.head&&r.head[0]?" | first: "+JSON.stringify(r.head[0]).slice(0,80):"");
    if(Array.isArray(r))return"["+r.length+" items]";
    var ks=Object.keys(r);
    return"{ "+ks.slice(0,6).join(", ")+(ks.length>6?" ...":"")+" }";
  }
  return String(r).slice(0,120);
}
var _resultIdCounter=0;
function buildToolPipelineHTML(data,opts){
  var evts=data.tool_events||[];
  var mix=data.tool_mix||{};
  if(!evts.length&&!Object.keys(mix).length)
    return'<p class="text-xs text-ink-400 italic">No tool call data found for this post.</p>';
  var hideTitle=!!(opts&&opts.hideTitle);
  var h='<div class="space-y-2">';
  if(!hideTitle){
    h+='<p class="text-xs font-bold uppercase tracking-widest text-ink-500 mb-2">';
    h+='&#9881; Tool Pipeline</p>';
  }
  var PHASE_LABELS={draft:"Draft Analysis",execution:"Execution"};
  var phases=[];var seen={};
  evts.forEach(function(ev){var p=ev.phase||"execution";if(!seen[p]){seen[p]=1;phases.push(p);}});
  var num=0;
  phases.forEach(function(phase){
    var phaseEvts=evts.filter(function(ev){return(ev.phase||"execution")===phase;});
    if(!phaseEvts.length)return;
    if(phases.length>1){
      h+='<div class="flex items-center gap-2 mt-3 mb-1">';
      h+='<span class="text-[10px] font-bold uppercase tracking-widest '+(phase==="draft"?"text-amber-600":"text-emerald-600")+'">'+(PHASE_LABELS[phase]||phase)+' ('+phaseEvts.length+')</span>';
      h+='<span class="flex-1 border-t border-ink-100/60"></span></div>';
    }
    var groups=groupParallel(phaseEvts);
    groups.forEach(function(grp){
    var isParallel=grp.length>1;
    if(isParallel){
      h+='<div class="border-l-2 border-blue-400 pl-3 space-y-2 relative">';
      h+='<span class="absolute -left-px top-0 text-[9px] font-bold text-blue-500 bg-white px-1 -translate-x-full">parallel ('+grp.length+')</span>';
    }
    grp.forEach(function(ev){
      num++;
      var cat=CAT_MAP[ev.tool]||"other";
      var cls=CAT_COLORS[cat]||CAT_COLORS.other;
      var elapsed=ev.elapsed_ms!=null?' <span class="font-mono text-ink-400">'+fmtMs(ev.elapsed_ms)+'</span>':"";
      var argLine=summarizeArgs(ev.args);
      var rp=ev.result||ev.result_preview;
      var preview=summarizeResult(rp);
      var err=ev.error?'<span class="text-red-500 text-[10px]">err: '+ev.error.slice(0,80)+'</span>':"";
      var rid="tool-result-"+(_resultIdCounter++);
      h+='<div class="flex items-start gap-2 text-xs">';
      h+='<span class="shrink-0 w-5 h-5 rounded-full bg-ink-100 text-ink-600 flex items-center justify-center font-bold text-[10px]">'+num+'</span>';
      h+='<div class="min-w-0 flex-1">';
      h+='<span class="inline-block rounded-full px-2 py-0.5 text-[10px] font-bold '+cls+'">'+ev.tool+'</span>';
      h+=elapsed;
      if(ev.source)h+=' <span class="text-[10px] text-ink-400">['+ev.source+']</span>';
      if(argLine)h+='<p class="text-ink-500 mt-0.5 truncate" title="'+argLine.replace(/"/g,"&quot;")+'">'+argLine+'</p>';
      if(preview)h+='<p class="text-ink-400 mt-0.5 truncate italic" title="'+preview.replace(/"/g,"&quot;")+'">&#8594; '+preview+'</p>';
      if(err)h+='<p class="mt-0.5">'+err+'</p>';
      if(rp){
        var fullJson;try{fullJson=JSON.stringify(rp,null,2);}catch(e){fullJson=String(rp);}
        h+='<button data-result-toggle="'+rid+'" class="text-[10px] text-ink-400 hover:text-brand-600 cursor-pointer mt-0.5">&#9656; result</button>';
        h+='<pre id="'+rid+'" class="hidden text-[10px] bg-ink-50 rounded p-2 mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all">'+fullJson.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")+'</pre>';
      }
      h+='</div></div>';
    });
    if(isParallel)h+='</div>';
    });
  });
  if(Object.keys(mix).length){
    h+='<div class="flex flex-wrap gap-2 pt-2 border-t border-ink-100/60 mt-2">';
    h+='<span class="text-[10px] font-bold uppercase tracking-widest text-ink-700">Mix:</span>';
    Object.keys(mix).forEach(function(k){
      var cat=k.toLowerCase();
      var cls=CAT_COLORS[cat]||CAT_COLORS.other;
      var dots="";for(var d=0;d<Math.min(mix[k],5);d++)dots+="\u25CF";
      h+='<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold '+cls+'">'+dots+' '+k+' ('+mix[k]+')</span>';
    });
    h+='</div>';
  }
  h+='</div>';
  return h;
}
function promptPhaseLabel(phase){
  var token=String(phase||"").toLowerCase();
  if(token==="draft")return"Draft Prompt";
  if(token==="execution")return"Execution Prompt";
  if(token==="board")return"Board Prompt";
  return token?token:"Prompt";
}
function buildPromptSectionHTML(title,body,meta){
  var text=String(body||"");
  if(!text)return"";
  var h='<div class="rounded-[18px] border border-ink-200/60 bg-white/75 p-4 shadow-sm">';
  h+='<p class="text-[10px] font-bold uppercase tracking-widest text-ink-900">'+escapeHtml(title)+'</p>';
  if(meta){
    h+='<p class="mt-1 text-[10px] text-ink-700">'+escapeHtml(meta)+'</p>';
  }
  h+='<pre class="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-[14px] bg-ink-950 px-3 py-3 text-[11px] leading-relaxed text-ink-50">'+escapeHtml(text)+'</pre>';
  h+='</div>';
  return h;
}
function buildPromptBundleHTML(data){
  var payload=data&&typeof data==="object"?data:{};
  var bundle=payload.prompt_bundle&&typeof payload.prompt_bundle==="object"?payload.prompt_bundle:{};
  var phases=Array.isArray(bundle.phases)?bundle.phases:[];
  var hasTools=Array.isArray(payload.tool_events)&&payload.tool_events.length;
  if(!bundle.system_prompt&&!phases.length&&!hasTools){
    return'<p class="text-xs text-ink-400 italic">No prompt bundle found for this post.</p>';
  }
  var h='<div class="space-y-3">';
  h+='<p class="text-xs font-bold uppercase tracking-widest text-ink-900">Prompt Details</p>';
  h+=buildPromptSectionHTML("System Prompt",bundle.system_prompt||"","");
  phases.forEach(function(item){
    if(!item||typeof item!=="object")return;
    var metaBits=[];
    if(item.session_id)metaBits.push("session "+String(item.session_id));
    if(item.resume_session)metaBits.push("resumed");
    h+=buildPromptSectionHTML(promptPhaseLabel(item.phase),item.prompt||"",metaBits.join(" · "));
  });
  if(payload.analysis_funnel&&Object.keys(payload.analysis_funnel).length){
    h+=buildPromptSectionHTML(
      "Analysis Funnel Snapshot",
      JSON.stringify(payload.analysis_funnel,null,2),
      "Pre-board funnel telemetry"
    );
  }
  if(hasTools){
    h+='<div class="rounded-[18px] border border-ink-200/60 bg-white/75 p-4 shadow-sm">';
    h+='<p class="text-[10px] font-bold uppercase tracking-widest text-ink-900">Compacted Tool Transcript</p>';
    h+='<div class="mt-2">'+buildToolPipelineHTML({tool_events:payload.tool_events||[],tool_mix:payload.tool_mix||{}},{hideTitle:true})+'</div>';
    h+='</div>';
  }
  h+='</div>';
  return h;
}
function fmtBoardTs(ts){
  if(!ts)return"";
  try{
    var d=new Date(ts);
    if(!isNaN(d)){
      return d.toLocaleString('ko-KR',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).replace(/\. /g,'-').replace('.','');
    }
  }catch(_e){}
  return String(ts).length>19?String(ts).substring(0,19).replace("T"," "):String(ts);
}
function thesisStateLabel(chain){
  var token=String(chain.terminal_event_type||chain.state||"").toLowerCase();
  var map={
    thesis_open:"Open",
    thesis_update:"Updated",
    thesis_invalidated:"Invalidated",
    thesis_realized:"Realized",
    open:"Open",
    updated:"Updated",
    invalidated:"Invalidated",
    realized:"Realized"
  };
  return map[token]||"Thesis";
}
function thesisStateBadge(chain){
  var token=String(chain.terminal_event_type||chain.state||"").toLowerCase();
  var cls="bg-slate-100 text-slate-700";
  if(token==="thesis_invalidated"||token==="invalidated")cls="bg-rose-100 text-rose-700";
  else if(token==="thesis_realized"||token==="realized")cls="bg-emerald-100 text-emerald-700";
  else if(token==="thesis_update"||token==="updated")cls="bg-amber-100 text-amber-700";
  else if(token==="thesis_open"||token==="open")cls="bg-sky-100 text-sky-700";
  return '<span class="inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold '+cls+'">'+thesisStateLabel(chain)+'</span>';
}
function thesisEventLabel(eventType){
  var map={
    thesis_open:"Open",
    thesis_update:"Update",
    thesis_invalidated:"Invalidated",
    thesis_realized:"Realized"
  };
  return map[String(eventType||"").toLowerCase()]||String(eventType||"");
}
function buildThesisPanelHTML(data){
  var chains=data&&Array.isArray(data.chains)?data.chains:[];
  if(!chains.length)return"";
  var h='<p class="text-xs font-bold uppercase tracking-widest text-ink-500 mb-2">Related Memory ('+chains.length+')</p>';
  h+='<div class="space-y-3">';
  chains.forEach(function(chain){
    var ticker=chain.ticker||"";
    var title=ticker?('<span class="font-display text-base font-bold tracking-tight text-ink-900">'+ticker+'</span>'):'<span class="font-display text-base font-bold tracking-tight text-ink-900">Thesis</span>';
    var summary=chain.thesis_summary||"Stored thesis chain";
    var refs=Array.isArray(chain.strategy_refs)?chain.strategy_refs:[];
    h+='<div class="rounded-[18px] border border-ink-200/60 bg-white/75 p-4 shadow-sm">';
    h+='<div class="flex items-start justify-between gap-3">';
    h+='<div class="min-w-0 flex-1"><div class="flex items-center gap-2">'+title;
    if(chain.side){h+='<span class="rounded-full bg-ink-100/80 px-2 py-0.5 text-[10px] font-bold text-ink-600">'+chain.side+'</span>';}
    h+='</div>';
    h+='<p class="mt-1 text-sm leading-relaxed text-ink-700">'+summary+'</p>';
    if(refs.length){
      h+='<div class="mt-2 flex flex-wrap gap-1.5">';
      refs.forEach(function(ref){
        h+='<span class="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700 ring-1 ring-amber-100">'+String(ref)+'</span>';
      });
      h+='</div>';
    }
    h+='</div>';
    h+=thesisStateBadge(chain);
    h+='</div>';
    if(chain.reflection&&chain.reflection.summary){
      h+='<div class="mt-3 rounded-[14px] border border-emerald-100 bg-emerald-50/60 px-3 py-3">';
      h+='<p class="text-[10px] font-bold uppercase tracking-widest text-emerald-700">Compacted Lesson</p>';
      h+='<p class="mt-1 text-xs leading-relaxed text-emerald-900">'+chain.reflection.summary+'</p>';
      if(chain.reflection.created_at){
        h+='<p class="mt-1 text-[10px] text-emerald-700/80">'+fmtBoardTs(chain.reflection.created_at)+'</p>';
      }
      h+='</div>';
    }
    if(Array.isArray(chain.events)&&chain.events.length){
      h+='<div class="mt-3 space-y-2">';
      chain.events.forEach(function(ev){
        h+='<div class="flex items-start gap-2 text-xs">';
        h+='<span class="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-300"></span>';
        h+='<div class="min-w-0 flex-1">';
        h+='<div class="flex flex-wrap items-center gap-2">';
        h+='<span class="font-semibold text-ink-700">'+thesisEventLabel(ev.event_type)+'</span>';
        if(ev.created_at){h+='<span class="text-[10px] text-ink-400">'+fmtBoardTs(ev.created_at)+'</span>';}
        h+='</div>';
        if(ev.summary){h+='<p class="mt-0.5 text-ink-600">'+ev.summary+'</p>';}
        h+='</div></div>';
      });
      h+='</div>';
    }
    h+='</div>';
  });
  h+='</div>';
  return h;
}
function setToggleLabel(btn,panel,label){
  if(!btn||!panel)return;
  btn.innerHTML=(panel.classList.contains("hidden")?"&#9656; ":"&#9662; ")+label;
}
document.addEventListener("click",function(ev){
  var btn=ev.target.closest("[data-prompt-toggle]");
  if(!btn)return;
  var article=btn.closest("article");
  var panel=article.querySelector("[data-prompt-panel]");
  if(panel.dataset.loaded==="1"){
    panel.classList.toggle("hidden");
    setToggleLabel(btn,panel,"Prompt Details");
    return;
  }
  btn.innerHTML="&#8987; Loading...";
  var url="/api/board/prompt?tenant_id="+encodeURIComponent(_boardTenant)
    +"&agent_id="+encodeURIComponent(article.dataset.agentId)
    +"&ts="+encodeURIComponent(article.dataset.ts);
  fetch(url).then(function(r){
    if(!r.ok)throw new Error("HTTP "+r.status);
    return r.json();
  }).then(function(data){
    if(data.error){
      panel.innerHTML='<p class="text-xs text-red-500">'+escapeHtml(data.error)+'</p>';
    }else{
      panel.innerHTML=buildPromptBundleHTML(data);
    }
    panel.dataset.loaded="1";
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Prompt Details");
  }).catch(function(e){
    panel.innerHTML='<p class="text-xs text-red-500">Failed to load prompt bundle: '+escapeHtml(e.message||e)+'</p>';
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Prompt Details");
  });
});
document.addEventListener("click",function(ev){
  var btn=ev.target.closest("[data-tool-toggle]");
  if(!btn)return;
  var article=btn.closest("article");
  var panel=article.querySelector("[data-tool-panel]");
  if(panel.dataset.loaded==="1"){
    panel.classList.toggle("hidden");
    setToggleLabel(btn,panel,"Tool Details");
    return;
  }
  btn.innerHTML="&#8987; Loading...";
  var url="/api/board/tools?tenant_id="+encodeURIComponent(_boardTenant)
    +"&agent_id="+encodeURIComponent(article.dataset.agentId)
    +"&ts="+encodeURIComponent(article.dataset.ts);
  fetch(url).then(function(r){
    if(!r.ok)throw new Error("HTTP "+r.status);
    return r.json();
  }).then(function(data){
    if(data.error){
      panel.innerHTML='<p class="text-xs text-red-500">'+data.error+'</p>';
    }else{
      panel.innerHTML=buildToolPipelineHTML(data);
    }
    panel.dataset.loaded="1";
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Tool Details");
  }).catch(function(e){
    panel.innerHTML='<p class="text-xs text-red-500">Failed to load tool data: '+escapeHtml(e.message||e)+'</p>';
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Tool Details");
  });
});
document.addEventListener("click",function(ev){
  var btn=ev.target.closest("[data-memory-toggle]");
  if(!btn)return;
  var article=btn.closest("article");
  var panel=article.querySelector("[data-theses-panel]");
  if(panel.dataset.loaded==="1"){
    panel.classList.toggle("hidden");
    setToggleLabel(btn,panel,"Related Memory");
    return;
  }
  btn.innerHTML="&#8987; Loading...";
  var cycleId=article.dataset.cycleId||"";
  var agentId=article.dataset.agentId||"";
  var url="/api/board/theses?tenant_id="+encodeURIComponent(_boardTenant)
    +"&cycle_id="+encodeURIComponent(cycleId)
    +"&agent_id="+encodeURIComponent(agentId);
  fetch(url).then(function(r){
    if(!r.ok)throw new Error("HTTP "+r.status);
    return r.json();
  }).then(function(data){
    var html=buildThesisPanelHTML(data);
    if(!html){
      panel.innerHTML='<p class="text-xs text-ink-400 italic">No related memory found for this post.</p>';
    }else{
      panel.innerHTML=html;
    }
    panel.dataset.loaded="1";
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Related Memory");
  }).catch(function(e){
    panel.innerHTML='<p class="text-xs text-red-500">Failed to load related memory: '+escapeHtml(e.message||e)+'</p>';
    panel.classList.remove("hidden");
    setToggleLabel(btn,panel,"Related Memory");
  });
});
document.addEventListener("click",function(ev){
  var btn=ev.target.closest("[data-result-toggle]");
  if(!btn)return;
  var id=btn.getAttribute("data-result-toggle");
  var pre=document.getElementById(id);
  if(!pre)return;
  pre.classList.toggle("hidden");
  btn.innerHTML=pre.classList.contains("hidden")?"&#9656; result":"&#9662; result";
});
document.addEventListener("DOMContentLoaded",function(){
  var articles=document.querySelectorAll("article[data-cycle-id]");
  articles.forEach(function(article){
    var panel=article.querySelector("[data-trades-panel]");
    if(!panel)return;
    var cycleId=article.dataset.cycleId||"";
    var agentId=article.dataset.agentId||"";
    if(!cycleId)return;
    var url="/api/board/trades?tenant_id="+encodeURIComponent(_boardTenant)
      +"&cycle_id="+encodeURIComponent(cycleId)
      +"&agent_id="+encodeURIComponent(agentId);
    fetch(url).then(function(r){
      if(!r.ok)throw new Error("HTTP "+r.status);
      return r.json();
    }).then(function(rows){
      if(!rows||!rows.length){
        panel.innerHTML="";
      }else{
        var h='<p class="text-xs font-bold uppercase tracking-widest text-ink-500 mb-2">Related Trades ('+rows.length+')</p>';
        h+='<div class="overflow-x-auto styled-scrollbar"><table class="min-w-full text-left text-xs">';
        h+='<thead><tr class="border-b border-ink-200/60 text-[10px] font-bold uppercase tracking-widest text-ink-400"><th class="px-2 py-1.5">Time</th><th class="px-2 py-1.5">Ticker</th><th class="px-2 py-1.5">Side</th><th class="px-2 py-1.5">ReqQty</th><th class="px-2 py-1.5">FillQty</th><th class="px-2 py-1.5">AvgPx</th><th class="px-2 py-1.5">Status</th></tr></thead><tbody>';
        rows.forEach(function(r){
          var side=r.side||"";
          var sideBadge=side==="BUY"?'<span class="inline-flex rounded-full bg-red-100/80 px-1.5 py-0.5 text-[10px] font-bold text-red-700">BUY</span>':side==="SELL"?'<span class="inline-flex rounded-full bg-blue-100/80 px-1.5 py-0.5 text-[10px] font-bold text-blue-700">SELL</span>':side;
          var st=r.status||"";
          var stBadge=st==="FILLED"?'<span class="inline-flex rounded-full bg-green-100/80 px-1.5 py-0.5 text-[10px] font-bold text-green-700">FILLED</span>':st==="REJECTED"?'<span class="inline-flex rounded-full bg-ink-200/80 px-1.5 py-0.5 text-[10px] font-bold text-ink-700">REJECTED</span>':st;
          var ts=fmtBoardTs(r.created_at||"");
          h+='<tr class="border-b border-ink-100/50 hover:bg-ink-100/40">';
          h+='<td class="px-2 py-1.5 font-mono text-ink-500">'+ts+'</td>';
          h+='<td class="px-2 py-1.5 font-bold">'+(r.ticker||"")+'</td>';
          h+='<td class="px-2 py-1.5">'+sideBadge+'</td>';
          h+='<td class="px-2 py-1.5 font-mono">'+Math.round(r.requested_qty||0)+'</td>';
          h+='<td class="px-2 py-1.5 font-mono">'+Math.round(r.filled_qty||0)+'</td>';
          h+='<td class="px-2 py-1.5 font-mono">'+(r.avg_price_krw!=null?Number(r.avg_price_krw).toLocaleString():"-")+'</td>';
          h+='<td class="px-2 py-1.5">'+stBadge+'</td>';
          h+='</tr>';
        });
        h+='</tbody></table></div>';
        panel.innerHTML=h;
        panel.classList.remove("hidden");
      }
    }).catch(function(e){
      panel.innerHTML='<p class="text-xs text-red-500">Failed to load trades: '+escapeHtml(e.message||e)+'</p>';
      panel.classList.remove("hidden");
    });
  });
});
"""
            + "</script>"
        )

        datepicker_js = (
            '<script>'
            'document.addEventListener("DOMContentLoaded",function(){'
            'var form=document.getElementById("board-filter-form");'
            'function ensureMask(){'
            "let mask=document.getElementById('board-loading-mask');"
            'if(mask){return mask;}'
            "mask=document.createElement('div');"
            "mask.id='board-loading-mask';"
            "mask.style.position='fixed';"
            "mask.style.inset='0';"
            "mask.style.display='none';"
            "mask.style.alignItems='center';"
            "mask.style.justifyContent='center';"
            "mask.style.background='rgba(15,23,42,0.28)';"
            "mask.style.backdropFilter='blur(2px)';"
            "mask.style.zIndex='9998';"
            "const card=document.createElement('div');"
            "card.style.display='inline-flex';"
            "card.style.alignItems='center';"
            "card.style.gap='10px';"
            "card.style.padding='12px 14px';"
            "card.style.border='1px solid rgba(148,163,184,0.35)';"
            "card.style.borderRadius='12px';"
            "card.style.background='rgba(255,255,255,0.96)';"
            "card.style.boxShadow='0 10px 24px rgba(15,23,42,0.22)';"
            "const spinner=document.createElement('span');"
            "spinner.style.width='16px';"
            "spinner.style.height='16px';"
            "spinner.style.border='2px solid #cbd5e1';"
            "spinner.style.borderTopColor='#0f172a';"
            "spinner.style.borderRadius='9999px';"
            "spinner.style.display='inline-block';"
            "spinner.style.animation='arenaSpin 0.8s linear infinite';"
            "const text=document.createElement('span');"
            "text.style.fontSize='13px';"
            "text.style.fontWeight='700';"
            "text.style.color='#0f172a';"
            "text.textContent='불러오는 중...';"
            'card.appendChild(spinner);'
            'card.appendChild(text);'
            'mask.appendChild(card);'
            "if(!document.getElementById('arena-spin-style')){"
            "const st=document.createElement('style');"
            "st.id='arena-spin-style';"
            "st.textContent='@keyframes arenaSpin{to{transform:rotate(360deg);}}';"
            'document.head.appendChild(st);'
            "}"
            'document.body.appendChild(mask);'
            'return mask;'
            '}'
            'function showMask(){'
            'var mask=ensureMask();'
            "mask.style.display='flex';"
            "document.body.style.cursor='wait';"
            '}'
            'if(form){'
            "form.addEventListener('submit',function(){"
            'showMask();'
            '});'
            '}'
            'if(typeof flatpickr==="undefined"){return;}'
            'flatpickr("#board-date-input",{dateFormat:"Y-m-d",theme:"airbnb",allowInput:false,onChange:function(sel,dateStr){'
            'if(!dateStr||!form){return;}'
            "if(form.dataset.submitting==='1'){return;}"
            "form.dataset.submitting='1';"
            'showMask();'
            'window.requestAnimationFrame(function(){form.submit();});'
            '}});'
            '});'
            '</script>'
        )

        header_datepicker = render_ui_template(
            "board_header_datepicker.jinja2",
            auth_enabled=deps.auth_enabled,
            tenant=tenant,
            date_value=date_filter or "",
            agent_id=selected_agent_id,
        )

        body = render_ui_template(
            "board_body.jinja2",
            posts=posts,
            empty_state_message=empty_state_message,
            page=page,
            prev_url=prev_url,
            next_url=next_url,
            prev_disabled=page <= 1,
            next_disabled=not has_next,
            tool_accordion_js=tool_accordion_js,
            datepicker_js=datepicker_js,
        )
        return deps.html_response(
            deps.tailwind_layout(
                "\uac8c\uc2dc\ud310",
                body,
                active="board",
                needs_datepicker=True,
                header_extra=header_datepicker,
                tenant=tenant,
                user=deps.current_user(request),
            ),
            max_age=30,
        )

    @app.get("/api/board")
    def api_board(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent id"),
        limit: int = Query(default=80, ge=1, le=400),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"board:{tenant}:{agent_key}:{limit}:0:{token or ''}",
            deps.fetch_board,
            tenant_id=tenant,
            limit=limit,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        return deps.json_response(rows, max_age=30)

    @app.get("/api/board/tools")
    def api_board_tools(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(..., description="agent id"),
        ts: str = Query(..., description="ISO timestamp of the board post"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/tools?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_tool_events_for_post(
            tenant_id=tenant,
            agent_id=agent_id.strip().lower(),
            ts_iso=ts,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/prompt")
    def api_board_prompt(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(..., description="agent id"),
        ts: str = Query(..., description="ISO timestamp of the board post"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/prompt?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_prompt_bundle_for_post(
            tenant_id=tenant,
            agent_id=agent_id.strip().lower(),
            ts_iso=ts,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/theses")
    def api_board_theses(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        agent_id: str = Query(default="", description="agent id for filtering"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/theses?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_theses_for_board_post(
            tenant_id=tenant,
            cycle_id=cycle_id.strip() or None,
            agent_id=agent_id.strip() or None,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/trades")
    def api_board_trades(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        agent_id: str = Query(default="", description="agent id for filtering"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/trades?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        rows = deps.fetch_trades_for_board_post(
            tenant_id=tenant,
            cycle_id=cycle_id.strip() or None,
            agent_id=agent_id.strip() or None,
        )
        return deps.json_response(rows, max_age=60)
