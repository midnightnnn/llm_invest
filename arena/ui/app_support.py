from __future__ import annotations

import html
import os
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from arena.providers import canonical_provider

_KST = ZoneInfo("Asia/Seoul")


def _coerce_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def fmt_ts(value: object) -> str:
    parsed = _coerce_datetime(value)
    if parsed is not None:
        return parsed.astimezone(_KST).strftime("%Y-%m-%d %H:%M KST")
    return html.escape(str(value or ""))


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


def to_date(value: object) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(_KST).date().isoformat()
    return str(value or "")[:10]


_AGENT_LOGO_SVGS = {
    "gpt": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 16 16"><path d="M14.949 6.547a3.94 3.94 0 0 0-.348-3.273 4.11 4.11 0 0 0-4.4-1.934A4.1 4.1 0 0 0 8.423.2 4.15 4.15 0 0 0 6.305.086a4.1 4.1 0 0 0-1.891.948 4.04 4.04 0 0 0-1.158 1.753 4.1 4.1 0 0 0-1.563.679A4 4 0 0 0 .554 4.72a3.99 3.99 0 0 0 .502 4.731 3.94 3.94 0 0 0 .346 3.274 4.11 4.11 0 0 0 4.402 1.933c.382.425.852.764 1.377.995.526.231 1.095.35 1.67.346 1.78.002 3.358-1.132 3.901-2.804a4.1 4.1 0 0 0 1.563-.68 4 4 0 0 0 1.14-1.253 3.99 3.99 0 0 0-.506-4.716m-6.097 8.406a3.05 3.05 0 0 1-1.945-.694l.096-.054 3.23-1.838a.53.53 0 0 0 .265-.455v-4.49l1.366.778q.02.011.025.035v3.722c-.003 1.653-1.361 2.992-3.037 2.996m-6.53-2.75a2.95 2.95 0 0 1-.36-2.01l.095.057L5.29 12.09a.53.53 0 0 0 .527 0l3.949-2.246v1.555a.05.05 0 0 1-.022.041L6.473 13.3c-1.454.826-3.311.335-4.15-1.098m-.85-6.94A3.02 3.02 0 0 1 3.07 3.949v3.785a.51.51 0 0 0 .262.451l3.93 2.237-1.366.779a.05.05 0 0 1-.048 0L2.585 9.342a2.98 2.98 0 0 1-1.113-4.094zm11.216 2.571L8.747 5.576l1.362-.776a.05.05 0 0 1 .048 0l3.265 1.86a3 3 0 0 1 1.173 1.207 2.96 2.96 0 0 1-.27 3.2 3.05 3.05 0 0 1-1.36.997V8.279a.52.52 0 0 0-.276-.445m1.36-2.015-.097-.057-3.226-1.855a.53.53 0 0 0-.53 0L6.249 6.153V4.598a.04.04 0 0 1 .019-.04L9.533 2.7a3.07 3.07 0 0 1 3.257.139c.474.325.843.778 1.066 1.303.223.526.289 1.103.191 1.664zM5.503 8.575 4.139 7.8a.05.05 0 0 1-.026-.037V4.049c0-.57.166-1.127.476-1.607s.752-.864 1.275-1.105a3.08 3.08 0 0 1 3.234.41l-.096.054-3.23 1.838a.53.53 0 0 0-.265.455zm.742-1.577 1.758-1 1.762 1v2l-1.755 1-1.762-1z" fill="#10a37f"/></svg>',
    "gemini": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"><defs><linearGradient id="gem" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#4285f4"/><stop offset="50%" stop-color="#9b72cb"/><stop offset="100%" stop-color="#d96570"/></linearGradient></defs><path d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81" fill="url(#gem)"/></svg>',
    "claude": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 16 16"><path d="m3.127 10.604 3.135-1.76.053-.153-.053-.085H6.11l-.525-.032-1.791-.048-1.554-.065-1.505-.08-.38-.081L0 7.832l.036-.234.32-.214.455.04 1.009.069 1.513.105 1.097.064 1.626.17h.259l.036-.105-.089-.065-.068-.064-1.566-1.062-1.695-1.121-.887-.646-.48-.327-.243-.306-.104-.67.435-.48.585.04.15.04.593.456 1.267.981 1.654 1.218.242.202.097-.068.012-.049-.109-.181-.9-1.626-.96-1.655-.428-.686-.113-.411a2 2 0 0 1-.068-.484l.496-.674L4.446 0l.662.089.279.242.411.94.666 1.48 1.033 2.014.302.597.162.553.06.17h.105v-.097l.085-1.134.157-1.392.154-1.792.052-.504.25-.605.497-.327.387.186.319.456-.045.294-.19 1.23-.37 1.93-.243 1.29h.142l.161-.16.654-.868 1.097-1.372.484-.545.565-.601.363-.287h.686l.505.751-.226.775-.707.895-.585.759-.839 1.13-.524.904.048.072.125-.012 1.897-.403 1.024-.186 1.223-.21.553.258.06.263-.218.536-1.307.323-1.533.307-2.284.54-.028.02.032.04 1.029.098.44.024h1.077l2.005.15.525.346.315.424-.053.323-.807.411-3.631-.863-.872-.218h-.12v.073l.726.71 1.331 1.202 1.667 1.55.084.383-.214.302-.226-.032-1.464-1.101-.565-.497-1.28-1.077h-.084v.113l.295.432 1.557 2.34.08.718-.112.234-.404.141-.444-.08-.911-1.28-.94-1.44-.759-1.291-.093.053-.448 4.821-.21.246-.484.186-.403-.307-.214-.496.214-.98.258-1.28.21-1.016.19-1.263.112-.42-.008-.028-.092.012-.953 1.307-1.448 1.957-1.146 1.227-.274.109-.477-.247.045-.44.266-.39 1.586-2.018.956-1.25.617-.723-.004-.105h-.036l-4.212 2.736-.75.096-.324-.302.04-.496.154-.162 1.267-.871z" fill="#d97757"/></svg>',
}

_PROVIDER_API_KEY_HELP_HTML = {
    "gpt": 'API 키 발급:<br><a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a>',
    "gemini": 'API 키 발급:<br><a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a>',
    "claude": 'API 키 발급:<br><a href="https://console.anthropic.com/settings/keys" target="_blank">console.anthropic.com/settings/keys</a>',
    "deepseek": 'API 키 발급:<br><a href="https://platform.deepseek.com/api_keys" target="_blank">platform.deepseek.com/api_keys</a>',
}


def agent_logo_svg(agent_id: str) -> str:
    key = agent_id.lower().split("-")[0].split("_")[0]
    return _AGENT_LOGO_SVGS.get(key, "")


def provider_api_key_help_html(provider: str) -> str:
    key = canonical_provider(provider) or str(provider or "").strip().lower()
    return _PROVIDER_API_KEY_HELP_HTML.get(
        key,
        "API 키는 provider별 설정 문서를 확인해 주세요.",
    )


RUN_STATUS_META: dict[str, dict[str, str]] = {
    "success": {
        "label": "정상 완료",
        "wrap": "border-emerald-200/70 bg-emerald-50/85",
        "badge": "bg-emerald-100 text-emerald-700",
        "text": "text-emerald-800",
    },
    "warning": {
        "label": "경고",
        "wrap": "border-amber-200/70 bg-amber-50/85",
        "badge": "bg-amber-100 text-amber-700",
        "text": "text-amber-900",
    },
    "blocked": {
        "label": "실행 중단",
        "wrap": "border-rose-200/70 bg-rose-50/90",
        "badge": "bg-rose-100 text-rose-700",
        "text": "text-rose-900",
    },
    "failed": {
        "label": "실행 실패",
        "wrap": "border-rose-200/70 bg-rose-50/90",
        "badge": "bg-rose-100 text-rose-700",
        "text": "text-rose-900",
    },
    "skipped": {
        "label": "건너뜀",
        "wrap": "border-sky-200/70 bg-sky-50/90",
        "badge": "bg-sky-100 text-sky-700",
        "text": "text-sky-900",
    },
    "running": {
        "label": "실행 중",
        "wrap": "border-indigo-200/70 bg-indigo-50/90",
        "badge": "bg-indigo-100 text-indigo-700",
        "text": "text-indigo-900",
    },
}

RUN_REASON_LABELS: dict[str, str] = {
    "reconciliation_failed": "실계좌와 AI 장부가 맞지 않아 거래를 중단했습니다.",
    "market_closed": "휴장일 또는 주말이라 실행을 건너뛰었습니다.",
    "schedule_closed": "예약된 실행 시간이 아니라 건너뛰었습니다.",
    "runtime_build_failed": "실행 준비 단계에서 설정 또는 자격 증명 문제가 발생했습니다.",
    "broker_order_rejected": "주문 반려가 있어 경고로 종료했습니다.",
    "cycle_report_error": "주문 처리 중 일부 오류가 발생했습니다.",
    "sync_failed": "시세 동기화 단계에서 오류가 발생했습니다.",
    "forecast_failed": "예측 계산 단계에서 오류가 발생했습니다.",
    "unexpected_exception": "예상하지 못한 오류로 실행이 중단됐습니다.",
    "system_exit": "실행이 중단되었습니다.",
}

RUN_STAGE_LABELS: dict[str, str] = {
    "runtime": "준비",
    "market_guard": "장 개장 확인",
    "schedule_guard": "스케줄 확인",
    "sync": "계좌/체결 동기화",
    "reconcile": "장부 대사",
    "research": "리서치",
    "trade": "주문",
    "memory_compaction": "메모리 정리",
    "forecast": "예측 계산",
    "agent_cycle": "에이전트 실행",
    "complete": "완료",
    "start": "시작",
}


def default_prompt_template(filename: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "agents" / "prompts" / filename
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
