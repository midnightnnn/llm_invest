from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Iterable
from uuid import uuid4

from arena.agents.base import TradingAgent
from arena.board.store import BoardStore
from arena.config import Settings
from arena.context import ContextBuilder
from arena.execution.gateway import ExecutionGateway
from arena.logging_utils import event_extra, failure_extra
from arena.market_sources import live_market_sources_for_markets, parse_markets
from arena.models import AccountSnapshot, BoardPost, ExecutionReport, ExecutionStatus, Position, Side, utc_now

logger = logging.getLogger(__name__)


class ArenaOrchestrator:
    """Coordinates multi-agent proposal, posting, and centralized execution."""

    def __init__(
        self,
        settings: Settings,
        context_builder: ContextBuilder,
        board_store: BoardStore,
        gateway: ExecutionGateway,
        agents: Iterable[TradingAgent],
    ):
        self.settings = settings
        self.context_builder = context_builder
        self.board_store = board_store
        self.gateway = gateway
        self.agents = list(agents)
        self.last_cycle_id: str = ""

    def _tenant_label(self) -> str:
        """Resolves tenant label for cycle diagnostics."""
        resolve = getattr(self.gateway.repo, "resolve_tenant_id", None)
        if callable(resolve):
            try:
                tenant = str(resolve() or "").strip().lower()
            except Exception:
                tenant = ""
            if tenant:
                return tenant
        return (os.getenv("ARENA_TENANT_ID", "") or "").strip().lower() or "-"

    def _append_runtime_warning(self, *, action: str, detail: dict[str, Any]) -> None:
        """Best-effort audit trail for degraded orchestrator paths."""
        append = getattr(self.gateway.repo, "append_runtime_audit_log", None)
        if not callable(append):
            return
        try:
            append(
                action=action,
                status="warning",
                tenant_id=self._tenant_label(),
                detail=detail,
            )
        except Exception as exc:
            logger.warning(
                "[yellow]Runtime audit append failed[/yellow] action=%s err=%s",
                action,
                str(exc),
            )

    def _virtual_total_cash_krw(self) -> float:
        """Returns total sleeve init capital (sum of per-agent capitals)."""
        if self.settings.agent_capitals:
            agent_ids = {a.agent_id for a in self.agents}
            return sum(
                v for k, v in self.settings.agent_capitals.items() if k in agent_ids
            ) or max(float(self.settings.sleeve_capital_krw), 0.0) * max(len(self.agents), 1)
        n = max(len(self.agents), 1)
        per_agent = max(float(self.settings.sleeve_capital_krw), 0.0)
        return per_agent * float(n)

    @staticmethod
    def _trim_text(text: Any, *, max_len: int) -> str:
        raw = str(text or "").replace("\n", " ").strip()
        if len(raw) > max_len:
            return raw[: max_len - 3] + "..."
        return raw

    def _board_context_from_posts(self, posts: list[dict[str, Any]]) -> str:
        """Builds compact board context from same-cycle shared explore posts.

        Uses ``draft_summary`` when available (agent-written concise summary).
        Falls back to truncated body when summary is missing.
        """
        if not posts:
            return ""
        lines: list[str] = ["[다른 에이전트 explore 요약 — 각 에이전트가 직접 작성한 핵심 요약입니다]"]
        limit = max(1, int(self.settings.context_max_board_posts))
        for row in posts[:limit]:
            aid = self._trim_text(row.get("agent_id"), max_len=24)
            summary = str(row.get("draft_summary") or "").strip()
            if summary:
                lines.append(f"[{aid}] {summary}")
            else:
                # Fallback: title + truncated body for agents that omit draft_summary.
                title = self._trim_text(row.get("title"), max_len=120)
                body = self._trim_text(row.get("body"), max_len=400)
                lines.append(f"[{aid}] [fallback:title+body] {title} | {body}")
        return "\n".join(lines)

    @staticmethod
    def _cycle_error_report(*, agent_id: str, message: str) -> ExecutionReport:
        """Builds a synthetic cycle-level error report when an agent stage aborts."""
        return ExecutionReport(
            status=ExecutionStatus.ERROR,
            order_id=f"cycle_err_{uuid4().hex[:10]}",
            filled_qty=0.0,
            avg_price_krw=0.0,
            avg_price_native=None,
            quote_currency="",
            fx_rate=0.0,
            message=f"agent={str(agent_id or '').strip() or '-'} {message}",
            created_at=utc_now(),
        )

    def _market_sources(self) -> list[str] | None:
        """Matches ContextBuilder market_sources so sleeve NAV uses same trusted sources."""
        if self.settings.trading_mode != "live":
            return None
        return live_market_sources_for_markets(parse_markets(self.settings.kis_target_market)) or None

    def _apply_fill(
        self,
        snapshot: AccountSnapshot,
        report: ExecutionReport,
        side: Side,
        ticker: str,
        price: float,
        exchange_code: str = "",
        instrument_id: str = "",
    ) -> None:
        """Applies a simulated/fill report to local snapshot state for same-cycle consistency."""
        if report.status.value not in {"SIMULATED", "FILLED"}:
            return

        notional = report.filled_qty * report.avg_price_krw
        pos = snapshot.positions.get(ticker)

        if side == Side.BUY:
            snapshot.cash_krw -= notional
            if pos:
                new_qty = pos.quantity + report.filled_qty
                new_cost = pos.avg_price_krw * pos.quantity + notional
                pos.quantity = new_qty
                pos.avg_price_krw = new_cost / new_qty if new_qty > 0 else pos.avg_price_krw
                pos.market_price_krw = price
                if exchange_code and not pos.exchange_code:
                    pos.exchange_code = exchange_code
                if instrument_id and not pos.instrument_id:
                    pos.instrument_id = instrument_id
            else:
                snapshot.positions[ticker] = Position(
                    ticker=ticker,
                    exchange_code=exchange_code,
                    instrument_id=instrument_id,
                    quantity=report.filled_qty,
                    avg_price_krw=report.avg_price_krw,
                    market_price_krw=price,
                )
        else:
            snapshot.cash_krw += notional
            if pos:
                pos.quantity = max(0.0, pos.quantity - report.filled_qty)
                pos.market_price_krw = price
                if pos.quantity == 0.0:
                    snapshot.positions.pop(ticker, None)

        snapshot.total_equity_krw = snapshot.cash_krw + sum(p.market_value_krw() for p in snapshot.positions.values())

    def _finalize_board_post(
        self,
        *,
        agent: TradingAgent,
        cycle_id: str,
        initial_post: BoardPost,
        intents: list[Any],
        reports: list[ExecutionReport],
    ) -> BoardPost:
        """Lets agents synthesize a fact-grounded board post after execution."""
        finalize = getattr(agent, "finalize_board_post", None)
        if callable(finalize):
            post = finalize(
                cycle_id=cycle_id,
                initial_post=initial_post,
                intents=intents,
                reports=reports,
            )
            if not isinstance(post, BoardPost):
                raise TypeError(f"board finalize returned {type(post).__name__}, expected BoardPost")
            return post
        return initial_post

    def _load_sleeves(self) -> tuple[dict[str, AccountSnapshot], dict[str, float], dict[str, dict[str, Any]]]:
        """Returns (agent_id -> sleeve snapshot, agent_id -> baseline equity)."""
        agent_ids = [a.agent_id for a in self.agents]
        total_cash = self._virtual_total_cash_krw()
        include_simulated = self.settings.trading_mode != "live"

        force_reinit = str(os.getenv("ARENA_FORCE_SLEEVE_REINIT", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
        excluded_tickers = list(getattr(self.settings, "reconcile_excluded_tickers", []))
        if force_reinit and hasattr(self.gateway.repo, "reinitialize_agent_state_checkpoints"):
            logger.warning(
                "[yellow]Force checkpoint reinit enabled[/yellow] env=ARENA_FORCE_SLEEVE_REINIT total_cash=%.0f",
                total_cash,
            )
            kwargs = {"agent_ids": agent_ids, "total_cash_krw": total_cash}
            if excluded_tickers:
                kwargs["excluded_tickers"] = excluded_tickers
            self.gateway.repo.reinitialize_agent_state_checkpoints(**kwargs)
        elif force_reinit and hasattr(self.gateway.repo, "reinitialize_agent_sleeves"):
            logger.warning(
                "[yellow]Force sleeve reinit enabled[/yellow] env=ARENA_FORCE_SLEEVE_REINIT total_cash=%.0f",
                total_cash,
            )
            kwargs = {"agent_ids": agent_ids, "total_cash_krw": total_cash}
            if excluded_tickers:
                kwargs["excluded_tickers"] = excluded_tickers
            self.gateway.repo.reinitialize_agent_sleeves(**kwargs)
        elif hasattr(self.gateway.repo, "ensure_agent_state_checkpoints"):
            kwargs = {
                "agent_ids": agent_ids,
                "total_cash_krw": total_cash,
                "capital_per_agent": self.settings.agent_capitals or None,
            }
            if excluded_tickers:
                kwargs["excluded_tickers"] = excluded_tickers
            self.gateway.repo.ensure_agent_state_checkpoints(**kwargs)
        else:
            kwargs = {
                "agent_ids": agent_ids,
                "total_cash_krw": total_cash,
                "capital_per_agent": self.settings.agent_capitals or None,
            }
            if excluded_tickers:
                kwargs["excluded_tickers"] = excluded_tickers
            self.gateway.repo.ensure_agent_sleeves(**kwargs)

        sleeves: dict[str, AccountSnapshot] = {}
        baselines: dict[str, float] = {}
        metas: dict[str, dict[str, Any]] = {}
        sources = self._market_sources()

        for aid in agent_ids:
            sleeve, baseline, meta = self.gateway.repo.build_agent_sleeve_snapshot(
                agent_id=aid,
                sources=sources,
                include_simulated=include_simulated,
            )
            sleeves[aid] = sleeve
            baselines[aid] = baseline
            metas[aid] = meta

        return sleeves, baselines, metas

    def run_cycle(self, snapshot: AccountSnapshot | None = None) -> list[ExecutionReport]:
        """Runs one full arena cycle and returns all execution reports."""
        cycle_id = f"cycle_{uuid4().hex[:12]}"
        self.last_cycle_id = cycle_id
        logger.info("[cyan]Cycle start[/cyan] cycle_id=%s", cycle_id)

        sleeves, baselines, metas = self._load_sleeves()

        # Default prompt injection excludes historical board posts.
        # Only same-cycle shared explore summaries are injected in execution phase.
        base_board_posts: list[dict[str, Any]] = []
        shared_board_posts: list[dict[str, Any]] = []
        tenant = self._tenant_label()
        cycle_health: dict[str, list[str]] = {
            "explore_failed_agents": [],
            "execution_failed_agents": [],
            "board_finalize_failed_agents": [],
            "board_publish_failed_agents": [],
            "nav_failed_agents": [],
        }
        share_explore_summary = len(self.agents) > 1
        if self.agents:
            logger.info(
                "[cyan]Cycle explore[/cyan] agents=%d share_summary=%s",
                len(self.agents),
                str(share_explore_summary).lower(),
            )

            def _explore_one(agent: TradingAgent) -> dict[str, Any]:
                sleeve = sleeves.get(agent.agent_id) or AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={})
                context = self.context_builder.build(
                    agent_id=agent.agent_id,
                    snapshot=sleeve,
                    sleeve_baseline_equity_krw=float(baselines.get(agent.agent_id) or 0.0),
                    sleeve_meta=metas.get(agent.agent_id) or {},
                    agent_config=self.settings.agent_configs.get(agent.agent_id),
                    cycle_id=cycle_id,
                )
                context["cycle_phase"] = "explore"
                context["cycle_id"] = cycle_id
                context["share_explore_summary"] = share_explore_summary
                context["board_posts"] = base_board_posts
                context["board_context"] = ""
                output = agent.generate(context)
                if not share_explore_summary:
                    return {}
                return output.board_post.model_dump(mode="json")

            explore_posts: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
                futures = {pool.submit(_explore_one, agent): agent.agent_id for agent in self.agents}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            explore_posts.append(result)
                    except Exception as exc:
                        aid = futures[future]
                        cycle_health["explore_failed_agents"].append(aid)
                        logger.exception(
                            "[red]Explore failed[/red] agent=%s err=%s",
                            aid,
                            exc,
                            extra=failure_extra(
                                "cycle_explore_failed",
                                exc,
                                tenant_id=tenant,
                                cycle_id=cycle_id,
                                phase="explore",
                                agent_id=aid,
                            ),
                        )

            shared_board_posts = list(explore_posts)
            try:
                shared_board_posts.sort(key=lambda p: str(p.get("created_at") or ""), reverse=True)
            except Exception:
                pass

        reports: list[ExecutionReport] = []
        logger.info(
            "[cyan]Cycle execution[/cyan] agents=%d",
            len(self.agents),
            extra=event_extra(
                "cycle_execution_start",
                tenant_id=tenant,
                cycle_id=cycle_id,
                agents=len(self.agents),
            ),
        )

        def _execute_one(agent: TradingAgent) -> tuple[BoardPost | None, list[ExecutionReport], dict[str, str]]:
            """Runs one agent's execution phase: generate → process → apply_fill."""
            diagnostics: dict[str, str] = {}
            sleeve = sleeves.get(agent.agent_id) or AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={})
            try:
                context = self.context_builder.build(
                    agent_id=agent.agent_id,
                    snapshot=sleeve,
                    sleeve_baseline_equity_krw=float(baselines.get(agent.agent_id) or 0.0),
                    sleeve_meta=metas.get(agent.agent_id) or {},
                    agent_config=self.settings.agent_configs.get(agent.agent_id),
                    cycle_id=cycle_id,
                )
                context["cycle_phase"] = "execution"
                context["cycle_id"] = cycle_id
                context["board_posts"] = shared_board_posts
                context["board_context"] = self._board_context_from_posts(shared_board_posts)
                output = agent.generate(context)
            except Exception as exc:
                diagnostics["execution_failed"] = str(exc)
                logger.exception(
                    "[red]Execution failed[/red] agent=%s err=%s",
                    agent.agent_id,
                    exc,
                    extra=failure_extra(
                        "cycle_execution_failed",
                        exc,
                        tenant_id=tenant,
                        cycle_id=cycle_id,
                        phase="execution",
                        agent_id=agent.agent_id,
                    ),
                )
                return None, [
                    self._cycle_error_report(
                        agent_id=agent.agent_id,
                        message=f"execution bootstrap failed: {exc}",
                    )
                ], diagnostics

            agent_reports: list[ExecutionReport] = []
            for intent in output.intents:
                try:
                    report = self.gateway.process(intent=intent, snapshot=sleeve)
                    agent_reports.append(report)
                    self._apply_fill(
                        snapshot=sleeve,
                        report=report,
                        side=intent.side,
                        ticker=intent.ticker,
                        price=float(intent.price_krw),
                        exchange_code=str(intent.exchange_code or ""),
                        instrument_id=str(intent.instrument_id or ""),
                    )
                except Exception as exc:
                    logger.exception(
                        "[red]Order execution failed[/red] agent=%s ticker=%s err=%s",
                        agent.agent_id,
                        intent.ticker,
                        exc,
                        extra=failure_extra(
                            "cycle_order_execution_failed",
                            exc,
                            tenant_id=tenant,
                            cycle_id=cycle_id,
                            phase="execution",
                            agent_id=agent.agent_id,
                            ticker=intent.ticker,
                            intent_id=intent.intent_id,
                            side=intent.side.value,
                        ),
                    )
                    agent_reports.append(
                        ExecutionReport(
                            status=ExecutionStatus.ERROR,
                            order_id=f"exec_{uuid4().hex[:10]}",
                            filled_qty=0.0,
                            avg_price_krw=0.0,
                            avg_price_native=None,
                            quote_currency=str(intent.quote_currency or ""),
                            fx_rate=float(intent.fx_rate or 0.0),
                            message=str(exc),
                            created_at=utc_now(),
                        )
                    )

            board_post: BoardPost | None = None
            try:
                board_post = self._finalize_board_post(
                    agent=agent,
                    cycle_id=cycle_id,
                    initial_post=output.board_post,
                    intents=list(output.intents),
                    reports=agent_reports,
                )
            except Exception as exc:
                diagnostics["board_finalize_failed"] = str(exc)
                logger.exception(
                    "[red]Board finalize failed[/red] agent=%s err=%s",
                    agent.agent_id,
                    exc,
                    extra=failure_extra(
                        "cycle_board_finalize_failed",
                        exc,
                        tenant_id=tenant,
                        cycle_id=cycle_id,
                        phase="board",
                        agent_id=agent.agent_id,
                    ),
                )
            if board_post is not None:
                try:
                    self.board_store.publish(board_post)
                    logger.info("[cyan]BOARD[/cyan] agent=%s title=%s", board_post.agent_id, board_post.title)
                except Exception as exc:
                    diagnostics["board_publish_failed"] = str(exc)
                    logger.exception(
                        "[yellow]Board publish failed[/yellow] agent=%s err=%s",
                        agent.agent_id,
                        exc,
                        extra=failure_extra(
                            "cycle_board_publish_failed",
                            exc,
                            tenant_id=tenant,
                            cycle_id=cycle_id,
                            phase="board",
                            agent_id=agent.agent_id,
                        ),
                    )
            return board_post, agent_reports, diagnostics

        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = {pool.submit(_execute_one, agent): agent.agent_id for agent in self.agents}
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    _board_post, agent_reports, diagnostics = future.result()
                    reports.extend(agent_reports)
                    if diagnostics.get("execution_failed"):
                        cycle_health["execution_failed_agents"].append(aid)
                    if diagnostics.get("board_finalize_failed"):
                        cycle_health["board_finalize_failed_agents"].append(aid)
                    if diagnostics.get("board_publish_failed"):
                        cycle_health["board_publish_failed_agents"].append(aid)
                except Exception as exc:
                    logger.exception(
                        "[red]Execution failed[/red] agent=%s err=%s",
                        aid,
                        exc,
                        extra=failure_extra(
                            "cycle_execution_worker_failed",
                            exc,
                            tenant_id=tenant,
                            cycle_id=cycle_id,
                            phase="execution",
                            agent_id=aid,
                        ),
                    )
                    cycle_health["execution_failed_agents"].append(aid)
                    reports.append(
                        self._cycle_error_report(
                            agent_id=aid,
                            message=f"execution worker crashed: {exc}",
                        )
                    )

        # Record per-agent NAV for performance comparisons.
        nav_date = utc_now().date()
        for agent in self.agents:
            sleeve = sleeves.get(agent.agent_id)
            if not sleeve:
                continue
            baseline = float(baselines.get(agent.agent_id) or 0.0)
            meta = metas.get(agent.agent_id) or {}
            try:
                self.gateway.repo.upsert_agent_nav_daily(
                    nav_date=nav_date,
                    agent_id=agent.agent_id,
                    nav_krw=float(sleeve.total_equity_krw),
                    baseline_equity_krw=baseline,
                    cash_krw=float(sleeve.cash_krw),
                    market_value_krw=sum(pos.market_value_krw() for pos in sleeve.positions.values()),
                    capital_flow_krw=float(meta.get("capital_flow_krw") or 0.0)
                    + float(meta.get("manual_cash_adjustment_krw") or 0.0),
                    fx_source=str(meta.get("fx_source") or ""),
                    valuation_source=str(meta.get("valuation_source") or "agent_sleeve_snapshot"),
                )
            except Exception as exc:
                cycle_health["nav_failed_agents"].append(agent.agent_id)
                logger.warning(
                    "[yellow]Agent NAV upsert failed[/yellow] agent=%s err=%s",
                    agent.agent_id,
                    str(exc),
                )

        error_reports = sum(1 for report in reports if report.status == ExecutionStatus.ERROR)
        if any(cycle_health.values()):
            logger.warning(
                "[yellow]Cycle finished with warnings[/yellow] cycle_id=%s reports=%d error_reports=%d explore_failed=%d execution_failed=%d board_finalize_failed=%d board_publish_failed=%d nav_failed=%d",
                cycle_id,
                len(reports),
                error_reports,
                len(cycle_health["explore_failed_agents"]),
                len(cycle_health["execution_failed_agents"]),
                len(cycle_health["board_finalize_failed_agents"]),
                len(cycle_health["board_publish_failed_agents"]),
                len(cycle_health["nav_failed_agents"]),
            )
            self._append_runtime_warning(
                action="orchestrator_cycle",
                detail={
                    "cycle_id": cycle_id,
                    "reports": len(reports),
                    "error_reports": error_reports,
                    "explore_failed_agents": cycle_health["explore_failed_agents"],
                    "execution_failed_agents": cycle_health["execution_failed_agents"],
                    "board_finalize_failed_agents": cycle_health["board_finalize_failed_agents"],
                    "board_publish_failed_agents": cycle_health["board_publish_failed_agents"],
                    "nav_failed_agents": cycle_health["nav_failed_agents"],
                },
            )
        else:
            logger.info("[green]Cycle finished[/green] cycle_id=%s reports=%d error_reports=%d", cycle_id, len(reports), error_reports)
        return reports
