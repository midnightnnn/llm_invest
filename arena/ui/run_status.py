from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RunStatusHelpers:
    latest_tenant_status_payload: Callable[[str], dict[str, Any] | None]
    header_status_kwargs: Callable[[str], dict[str, str]]


def build_run_status_helpers(
    *,
    repo: Any,
    cached_fetch: Callable[..., Any],
    parse_json_object: Callable[[object], dict[str, Any]],
    safe_float: Callable[[object, float], float],
    run_status_meta: dict[str, dict[str, str]],
    run_reason_labels: dict[str, str],
    run_stage_labels: dict[str, str],
) -> RunStatusHelpers:
    status_color_map: dict[str, str] = {
        "success": "emerald",
        "skipped": "sky",
        "running": "indigo",
        "warning": "amber",
        "blocked": "rose",
        "failed": "rose",
    }

    def coerce_json_object(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        return parse_json_object(raw)

    def format_reconciliation_issue(issue: dict[str, Any]) -> str:
        issue_type = str(issue.get("issue_type") or "").strip().lower()
        entity_key = str(issue.get("entity_key") or "").strip().upper()
        expected = coerce_json_object(issue.get("expected_json"))
        actual = coerce_json_object(issue.get("actual_json"))
        detail = coerce_json_object(issue.get("detail_json"))
        if issue_type == "position_quantity_mismatch":
            ledger_qty = safe_float(expected.get("ledger_quantity"), 0.0)
            broker_qty = safe_float(actual.get("broker_quantity"), 0.0)
            return f"{entity_key}: AI {ledger_qty:g}주, 실계좌 {broker_qty:g}주"
        if issue_type == "negative_agent_cash":
            return f"{entity_key}: 에이전트 현금이 음수로 계산됐습니다"
        if issue_type == "broker_cash_unallocated":
            broker_cash = safe_float(expected.get("broker_cash_krw"), 0.0)
            derived = safe_float(actual.get("derived_agent_cash_krw"), 0.0)
            return f"현금: 실계좌 {broker_cash:,.0f}원, AI 장부 {derived:,.0f}원"
        if issue_type.startswith("external_broker_position"):
            qty = safe_float(detail.get("excluded_quantity") or actual.get("broker_quantity"), 0.0)
            return f"{entity_key}: 외부 보유 {qty:g}주를 AI 계산에서 제외했습니다"
        if issue_type == "external_broker_trade_excluded":
            qty = safe_float(detail.get("excluded_quantity"), 0.0)
            return f"{entity_key}: AI 주문과 무관한 체결 {qty:g}주를 제외했습니다"
        return f"{entity_key or issue_type}: {str(issue.get('issue_type') or '').strip()}"

    def latest_tenant_status_payload(tenant: str) -> dict[str, Any] | None:
        fetch_status = getattr(repo, "latest_tenant_run_status", None)
        if not callable(fetch_status):
            return None
        try:
            row = fetch_status(tenant_id=tenant, exclude_statuses=["skipped"])
        except Exception:
            row = None
        if not row:
            try:
                row = fetch_status(tenant_id=tenant)
            except Exception:
                return None
        if not row:
            return None
        status = str(row.get("status") or "").strip().lower() or "unknown"
        reason_code = str(row.get("reason_code") or "").strip().lower()
        stage = str(row.get("stage") or "").strip().lower()
        detail = coerce_json_object(row.get("detail_json"))
        message = str(row.get("message") or "").strip() or run_reason_labels.get(reason_code) or "최근 실행 상태를 확인하세요."
        meta = dict(run_status_meta.get(status) or run_status_meta["warning"])
        issues: list[str] = []
        if status in {"blocked", "warning"} and reason_code == "reconciliation_failed":
            latest_recon = getattr(repo, "latest_reconciliation_run", None)
            if callable(latest_recon):
                try:
                    recon_row = latest_recon(tenant_id=tenant)
                except Exception:
                    recon_row = None
                if recon_row:
                    run_id = str(recon_row.get("run_id") or "").strip()
                    if run_id:
                        try:
                            issue_rows = repo.fetch_rows(
                                f"""
                                SELECT issue_type, entity_key, expected_json, actual_json, detail_json
                                FROM `{repo.dataset_fqn}.reconciliation_issues`
                                WHERE tenant_id = @tenant_id
                                  AND run_id = @run_id
                                  AND severity IN ('error', 'warning')
                                ORDER BY created_at ASC
                                LIMIT 3
                                """,
                                {"tenant_id": tenant, "run_id": run_id},
                            )
                        except Exception:
                            issue_rows = []
                        issues = [format_reconciliation_issue(row) for row in issue_rows]
        return {
            "status": status,
            "status_label": meta["label"],
            "status_wrap_class": meta["wrap"],
            "status_badge_class": meta["badge"],
            "status_text_class": meta["text"],
            "reason_code": reason_code,
            "reason_label": run_reason_labels.get(reason_code) or "",
            "stage": stage,
            "stage_label": run_stage_labels.get(stage) or (stage or "-"),
            "run_type": str(row.get("run_type") or "").strip().lower(),
            "message": message,
            "recorded_at": row.get("recorded_at"),
            "finished_at": row.get("finished_at"),
            "execution_name": str(row.get("execution_name") or "").strip(),
            "job_name": str(row.get("job_name") or "").strip(),
            "log_uri": str(row.get("log_uri") or "").strip(),
            "detail": detail,
            "issues": issues,
        }

    def header_status_kwargs(tenant: str) -> dict[str, str]:
        payload = cached_fetch(f"tenant_run_status:{tenant}", latest_tenant_status_payload, tenant)
        if not payload:
            return {}
        status = str(payload.get("status") or "")
        label = str(payload.get("status_label") or "")
        return {
            "status_label": label,
            "status_color": status_color_map.get(status, "emerald"),
        }

    return RunStatusHelpers(
        latest_tenant_status_payload=latest_tenant_status_payload,
        header_status_kwargs=header_status_kwargs,
    )
