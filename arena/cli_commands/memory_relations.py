from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


async def _run_relation_extraction_for_tenant(
    *,
    settings,
    repo,
    tenant_id: str,
    limit: int,
    source_table: str,
    event_types: list[str],
    dry_run: bool,
    provider: str,
    model: str,
    min_confidence: float,
    max_triples_per_source: int,
) -> list[dict[str, Any]]:
    from arena.memory.semantic_extractor import SemanticRelationExtractor

    extractor = SemanticRelationExtractor(
        settings=settings,
        repo=repo,
        provider=provider or None,
        model=model or None,
        min_confidence=min_confidence,
        max_triples_per_source=max_triples_per_source,
    )
    return await extractor.run_pending(
        tenant_id=tenant_id,
        limit=limit,
        source_table=source_table or None,
        event_types=event_types or None,
        dry_run=dry_run,
    )


def cmd_extract_memory_relations(
    *,
    tenant_ids: list[str] | None = None,
    limit: int = 25,
    source_table: str = "",
    event_types: list[str] | None = None,
    dry_run: bool = False,
    timeout_seconds: int = 0,
    provider: str = "",
    model: str = "",
    min_confidence: float = 0.65,
    max_triples_per_source: int = 6,
) -> None:
    """Runs semantic relation extraction as an offline memory maintenance job."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)

    if timeout_seconds > 0:
        settings.llm_timeout_seconds = int(timeout_seconds)

    settings.trading_mode = getattr(settings, "trading_mode", "paper").strip().lower()
    cli._validate_or_exit(settings, require_kis=False, require_llm=True, live=False)

    clean_source_table = str(source_table or "").strip()
    allowed_sources = {"", "agent_memory_events", "board_posts", "research_briefings"}
    if clean_source_table not in allowed_sources:
        logger.error("[red]Relation extraction failed[/red] unsupported source_table=%s", clean_source_table)
        raise SystemExit(2)

    clean_event_types: list[str] = []
    for token in event_types or []:
        event_type = str(token or "").strip()
        if event_type and event_type not in clean_event_types:
            clean_event_types.append(event_type)

    tenants: list[str] = []
    for token in tenant_ids or []:
        for tenant in cli._parse_tenant_tokens(token):
            if tenant not in tenants:
                tenants.append(tenant)
    if not tenants:
        active = str(cli._tenant_id() or "").strip().lower()
        tenants = [active or "local"]

    summary: list[dict[str, Any]] = []
    for tenant in tenants:
        repo = cli._repo_or_exit(settings, tenant_id=tenant)
        repo.ensure_dataset()
        repo.ensure_tables()
        logger.info(
            "[bold]Relation extraction start[/bold] tenant=%s limit=%d source=%s dry_run=%s",
            tenant,
            int(limit),
            clean_source_table or "all",
            "true" if dry_run else "false",
        )
        try:
            rows = asyncio.run(
                _run_relation_extraction_for_tenant(
                    settings=settings,
                    repo=repo,
                    tenant_id=tenant,
                    limit=max(1, int(limit)),
                    source_table=clean_source_table,
                    event_types=clean_event_types,
                    dry_run=dry_run,
                    provider=str(provider or "").strip(),
                    model=str(model or "").strip(),
                    min_confidence=float(min_confidence),
                    max_triples_per_source=max_triples_per_source,
                )
            )
        except Exception as exc:
            logger.exception("[red]Relation extraction failed[/red] tenant=%s err=%s", tenant, str(exc))
            raise SystemExit(1)
        accepted = sum(int(row.get("accepted_count") or 0) for row in rows)
        rejected = sum(int(row.get("rejected_count") or 0) for row in rows)
        logger.info(
            "[bold green]Relation extraction ok[/bold green] tenant=%s sources=%d accepted=%d rejected=%d dry_run=%s",
            tenant,
            len(rows),
            accepted,
            rejected,
            "true" if dry_run else "false",
        )
        summary.append(
            {
                "tenant_id": tenant,
                "sources": len(rows),
                "accepted_count": accepted,
                "rejected_count": rejected,
                "dry_run": bool(dry_run),
                "rows": rows,
            }
        )

    print(json.dumps({"result": summary}, ensure_ascii=False, indent=2, default=str))
