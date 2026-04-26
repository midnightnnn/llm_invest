"""Admin runtime ops: sleeve/NAV sync and remove-confirm warnings.

Triggered when an admin edits agent configuration; keeps live broker state
aligned with the new ``agents_config`` and surfaces user-facing warnings
before destructive removes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from arena.config import Settings
from arena.market_sources import live_market_sources_for_markets
from arena.providers import canonical_provider, provider_alias_map, runtime_row_api_key_status

logger = logging.getLogger(__name__)


def live_market_sources_for_market_value(
    *,
    tenant_settings: Settings,
    market_value: str,
    is_live_mode: Callable[[Settings | None], bool],
    live_market_sources: Callable[[Settings | None], list[str] | None],
) -> list[str] | None:
    if not is_live_mode(tenant_settings):
        return None
    sources = live_market_sources_for_markets(market_value)
    if sources:
        return sources
    return live_market_sources(tenant_settings)


@dataclass(frozen=True)
class AdminRuntimeOps:
    """Sync agent sleeves/NAV and build remove-confirm warnings."""

    repo: Any
    is_live_mode: Callable[[Settings | None], bool]
    live_market_sources: Callable[[Settings | None], list[str] | None]
    safe_float: Callable[[object, float], float]

    def sync_runtime_state(
        self,
        *,
        tenant: str,
        tenant_settings: Settings,
        entries: list[dict[str, Any]],
        updated_by: str,
        sources: list[str] | None,
    ) -> dict[str, Any]:
        parsed_agent_ids = [
            str(entry.get("id") or "").strip().lower()
            for entry in entries
            if str(entry.get("id") or "").strip()
        ]
        sync_summary: dict[str, Any] = {"agents": len(parsed_agent_ids)}
        if not parsed_agent_ids:
            return sync_summary

        capitals = [
            self.safe_float(entry.get("capital_krw"), tenant_settings.sleeve_capital_krw)
            for entry in entries
        ]
        avg_capital = sum(capitals) / len(capitals) if capitals else tenant_settings.sleeve_capital_krw
        target_capitals = {
            str(entry.get("id") or "").strip().lower(): self.safe_float(
                entry.get("capital_krw"), tenant_settings.sleeve_capital_krw
            )
            for entry in entries
            if str(entry.get("id") or "").strip()
        }

        tenant_is_live = self.is_live_mode(tenant_settings)
        sync_capitals = getattr(self.repo, "retarget_agent_capitals_preserve_positions", None)
        sync_sleeves = getattr(self.repo, "retarget_agent_sleeves_preserve_positions", None)
        sync_fn = sync_capitals if callable(sync_capitals) else sync_sleeves
        if callable(sync_fn):
            sync_kwargs = {
                "agent_ids": parsed_agent_ids,
                "target_sleeve_capital_krw": avg_capital,
                "target_capitals": target_capitals,
                "include_simulated": not tenant_is_live,
                "sources": sources,
                "tenant_id": tenant,
            }
            if sync_fn is sync_capitals:
                sync_kwargs["created_by"] = updated_by
            try:
                sync_out = sync_fn(**sync_kwargs)
                over_target_agents = sorted(
                    [
                        aid
                        for aid, meta in dict(sync_out or {}).items()
                        if bool(dict(meta).get("over_target"))
                    ]
                )
                sync_summary["over_target_agents"] = over_target_agents
            except Exception as retarget_exc:
                logger.warning(
                    "[yellow]Sleeve retarget after agents save failed[/yellow] tenant=%s err=%s",
                    tenant,
                    str(retarget_exc),
                )
                sync_summary["retarget_error"] = str(retarget_exc)

        build_sleeve = getattr(self.repo, "build_agent_sleeve_snapshot", None)
        upsert_nav = getattr(self.repo, "upsert_agent_nav_daily", None)
        if callable(build_sleeve) and callable(upsert_nav):
            nav_date = datetime.now(timezone.utc).date()
            nav_updated = 0
            nav_failed: list[str] = []
            for aid in parsed_agent_ids:
                try:
                    snap, baseline, meta = build_sleeve(
                        agent_id=aid,
                        sources=sources,
                        include_simulated=not tenant_is_live,
                        tenant_id=tenant,
                    )
                    upsert_nav(
                        nav_date=nav_date,
                        agent_id=aid,
                        nav_krw=float(snap.total_equity_krw),
                        baseline_equity_krw=float(baseline),
                        cash_krw=float(snap.cash_krw),
                        market_value_krw=sum(pos.market_value_krw() for pos in snap.positions.values()),
                        capital_flow_krw=float((meta or {}).get("capital_flow_krw") or 0.0)
                        + float((meta or {}).get("manual_cash_adjustment_krw") or 0.0),
                        fx_source=str((meta or {}).get("fx_source") or ""),
                        valuation_source=str((meta or {}).get("valuation_source") or "agent_sleeve_snapshot"),
                        tenant_id=tenant,
                    )
                    nav_updated += 1
                except Exception as nav_exc:
                    nav_failed.append(aid)
                    logger.warning(
                        "[yellow]Agent NAV sync after agents save failed[/yellow] tenant=%s agent=%s err=%s",
                        tenant,
                        aid,
                        str(nav_exc),
                    )
            sync_summary["nav_sync"] = {
                "updated": nav_updated,
                "failed_agents": nav_failed,
                "nav_date": nav_date.isoformat(),
            }
        return sync_summary

    def build_remove_warning(
        self,
        *,
        tenant: str,
        tenant_settings: Settings,
        agent_entry: dict[str, Any],
    ) -> str:
        aid = str(agent_entry.get("id") or "").strip().lower()
        provider = canonical_provider(str(agent_entry.get("provider") or "").strip().lower())
        if not provider:
            provider = provider_alias_map().get(aid, "")

        has_api_key = False
        latest_creds_loader = getattr(self.repo, "latest_runtime_credentials", None)
        if callable(latest_creds_loader):
            try:
                latest_creds = latest_creds_loader(tenant_id=tenant) or {}
                has_api_key = bool(runtime_row_api_key_status(latest_creds).get(provider))
            except Exception:
                has_api_key = False

        tenant_is_live = self.is_live_mode(tenant_settings)
        tenant_live_sources = self.live_market_sources(tenant_settings) if tenant_is_live else None
        sleeve_equity = 0.0
        baseline_equity = 0.0
        snapshot_available = False
        build_sleeve = getattr(self.repo, "build_agent_sleeve_snapshot", None)
        if callable(build_sleeve):
            try:
                snapshot, baseline, _meta = build_sleeve(
                    agent_id=aid,
                    sources=tenant_live_sources,
                    include_simulated=not tenant_is_live,
                    tenant_id=tenant,
                )
                sleeve_equity = float(getattr(snapshot, "total_equity_krw", 0.0) or 0.0)
                baseline_equity = float(baseline or 0.0)
                snapshot_available = True
            except Exception as exc:
                logger.warning(
                    "[yellow]Agent remove warning preflight failed[/yellow] tenant=%s agent=%s err=%s",
                    tenant,
                    aid,
                    str(exc),
                )

        warning_bits: list[str] = []
        if has_api_key:
            warning_bits.append("저장된 API key")

        active_equity = max(sleeve_equity, baseline_equity)
        if active_equity > 0:
            warning_bits.append(f"활성 슬리브 자금/평가금액 ₩{int(round(active_equity)):,}")
        elif not snapshot_available:
            configured_capital = self.safe_float(agent_entry.get("capital_krw"), 0.0)
            if configured_capital > 0:
                warning_bits.append(f"설정된 자본금 ₩{int(round(configured_capital)):,}")

        if not warning_bits:
            return ""

        message = f"'{aid}' 제거 전 확인: " + ", ".join(warning_bits) + " 이 있습니다. 정말 제거할까요?"
        if has_api_key:
            message += " 저장된 API key 자체는 삭제되지 않습니다."
        return message
