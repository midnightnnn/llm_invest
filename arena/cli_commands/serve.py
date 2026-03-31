from __future__ import annotations

import asyncio
import json
import logging
from datetime import date

from arena.config import Settings

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def cmd_serve_strategy_mcp() -> None:
    """Runs the strategy reference MCP server."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    logger.info("[bold]Starting MCP server[/bold] name=llm-arena-strategy-reference")
    cli.serve_mcp()


def cmd_serve_ui() -> None:
    """Serves a read-only UI for board/memory/NAV/trades."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)

    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()

    from arena.ui.server import serve_ui

    serve_ui(repo=repo, settings=settings)


def _provider_credentials_ready(provider: str, settings: Settings) -> bool:
    """Returns True when the requested provider has enough credentials for a smoke run."""
    from arena.agents.adk_agents import _has_credentials

    return _has_credentials(provider, settings)


async def _run_research_smoke(*, provider: str, prompt: str, settings: Settings) -> str:
    """Runs one standalone google_search-grounded research pass for a specific provider."""
    from google.adk import Agent, Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import google_search
    from google.genai import types

    from arena.agents.adk_agents import _resolve_model

    clean_provider = str(provider or "").strip().lower()
    if clean_provider not in {"gpt", "gemini", "claude"}:
        raise ValueError(f"Unsupported provider: {provider}")

    model_override = ""
    if clean_provider == "gemini":
        model_override = str(settings.research_gemini_model or "").strip()
    model = _resolve_model(clean_provider, settings, model_override=model_override)
    agent = Agent(
        name=f"research_smoke_{clean_provider}",
        model=model,
        instruction=(
            "You are a financial research smoke-test agent. "
            "Use Google Search when needed, then return a concise factual market summary."
        ),
        tools=[google_search],
    )
    session_service = InMemorySessionService()
    app_name = f"llm_arena_research_smoke_{clean_provider}"
    session_id = f"research_smoke_{clean_provider}_{int(date.today().strftime('%Y%m%d'))}"
    await session_service.create_session(
        app_name=app_name,
        user_id="arena",
        session_id=session_id,
    )
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    text = ""
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    async for event in runner.run_async(
        user_id="arena",
        session_id=session_id,
        new_message=message,
    ):
        if not event.content:
            continue
        for part in event.content.parts:
            if getattr(part, "text", None):
                text += part.text
    return text.strip()


async def _run_thesis_compaction_smoke(
    *,
    cycle_id: str,
    agent_ids: list[str],
    settings: Settings,
    repo,
    save: bool = False,
) -> list[dict[str, object]]:
    """Runs thesis-chain compaction without a full agent cycle."""
    from arena.agents.memory_compaction_agent import MemoryCompactionAgent
    from arena.memory.store import MemoryStore

    memory_store = MemoryStore(repo, trading_mode=settings.trading_mode, memory_policy=settings.memory_policy)
    compactor = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)
    clean_cycle_id = str(cycle_id or "").strip()
    clean_agent_ids: list[str] = []
    for token in agent_ids:
        agent_id = str(token or "").strip()
        if agent_id and agent_id not in clean_agent_ids:
            clean_agent_ids.append(agent_id)

    if save:
        saved = await compactor.run(cycle_id=clean_cycle_id, agent_ids=clean_agent_ids)
        return [{"agent_id": str(row.get("agent_id") or ""), "saved": row} for row in saved]

    preview: list[dict[str, object]] = []
    for agent_id in clean_agent_ids:
        inputs = compactor._load_agent_inputs(agent_id, clean_cycle_id)
        reflections = await compactor._compact_one(agent_id=agent_id, cycle_id=clean_cycle_id, inputs=inputs)
        preview.append(
            {
                "agent_id": agent_id,
                "closed_thesis_chains": inputs.get("closed_thesis_chains") or [],
                "cycle_memories": inputs.get("cycle_memories") or [],
                "board_posts": inputs.get("board_posts") or [],
                "environment_research": inputs.get("environment_research") or [],
                "prior_lessons": inputs.get("prior_lessons") or [],
                "reflections": reflections,
            }
        )
    return preview


def cmd_smoke_research(provider: str, *, prompt: str = "", timeout_seconds: int = 0) -> None:
    """Runs a narrow smoke test for provider + google_search research compatibility."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)

    clean_provider = str(provider or "").strip().lower()
    if not cli._provider_credentials_ready(clean_provider, settings):
        logger.error("[red]Research smoke failed[/red] missing credentials for provider=%s", clean_provider)
        raise SystemExit(2)

    if timeout_seconds > 0:
        settings.llm_timeout_seconds = int(timeout_seconds)

    smoke_prompt = (
        str(prompt or "").strip()
        or "Summarize the most important macro, rates, and U.S. equity market developments from the last 24 hours for an investor. Keep it concise."
    )

    logger.info("[bold]Research smoke start[/bold] provider=%s timeout=%ds", clean_provider, settings.llm_timeout_seconds)
    try:
        text = asyncio.run(cli._run_research_smoke(provider=clean_provider, prompt=smoke_prompt, settings=settings))
    except Exception as exc:
        logger.exception("[red]Research smoke failed[/red] provider=%s err=%s", clean_provider, str(exc))
        raise SystemExit(1)

    if not text:
        logger.error("[red]Research smoke failed[/red] provider=%s empty_response=true", clean_provider)
        raise SystemExit(1)

    logger.info("[bold green]Research smoke ok[/bold green] provider=%s chars=%d", clean_provider, len(text))
    print(text)


def cmd_smoke_thesis_compaction(
    cycle_id: str,
    *,
    agent_ids: list[str] | None = None,
    timeout_seconds: int = 0,
    save: bool = False,
) -> None:
    """Runs thesis-chain compaction directly for one cycle without the full agent loop."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)

    clean_cycle_id = str(cycle_id or "").strip()
    if not clean_cycle_id:
        logger.error("[red]Thesis compaction smoke failed[/red] cycle_id is required")
        raise SystemExit(2)

    if timeout_seconds > 0:
        settings.llm_timeout_seconds = int(timeout_seconds)

    settings.trading_mode = str(getattr(settings, "trading_mode", "paper") or "paper").strip().lower() or "paper"
    cli._validate_or_exit(settings, require_kis=False, require_llm=True, live=False)

    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()

    clean_agent_ids: list[str] = []
    for token in list(agent_ids or []) or list(getattr(settings, "agent_ids", []) or []):
        agent_id = str(token or "").strip()
        if agent_id and agent_id not in clean_agent_ids:
            clean_agent_ids.append(agent_id)
    if not clean_agent_ids:
        logger.error("[red]Thesis compaction smoke failed[/red] no agent ids configured")
        raise SystemExit(2)

    mode = "save" if save else "preview"
    logger.info(
        "[bold]Thesis compaction smoke start[/bold] cycle_id=%s agents=%s mode=%s timeout=%ds",
        clean_cycle_id,
        ",".join(clean_agent_ids),
        mode,
        settings.llm_timeout_seconds,
    )
    try:
        result = asyncio.run(
            cli._run_thesis_compaction_smoke(
                cycle_id=clean_cycle_id,
                agent_ids=clean_agent_ids,
                settings=settings,
                repo=repo,
                save=save,
            )
        )
    except Exception as exc:
        logger.exception(
            "[red]Thesis compaction smoke failed[/red] cycle_id=%s mode=%s err=%s",
            clean_cycle_id,
            mode,
            str(exc),
        )
        raise SystemExit(1)

    logger.info(
        "[bold green]Thesis compaction smoke ok[/bold green] cycle_id=%s agents=%d mode=%s",
        clean_cycle_id,
        len(clean_agent_ids),
        mode,
    )
    print(
        json.dumps(
            {
                "cycle_id": clean_cycle_id,
                "mode": mode,
                "agent_ids": clean_agent_ids,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
