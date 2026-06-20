from __future__ import annotations

from datetime import datetime, timezone

import arena.cli as cli
from arena.cli import build_parser
from arena.config import load_settings


def test_backfill_watch_items_parser_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "backfill-watch-items",
            "--tenant",
            "local",
            "--agent",
            "gpt",
            "--dry-run",
            "--include-existing",
            "--limit-per-agent",
            "250",
        ]
    )

    assert args.command == "backfill-watch-items"
    assert args.tenant == ["local"]
    assert args.agent == ["gpt"]
    assert args.dry_run is True
    assert args.include_existing is True
    assert args.limit_per_agent == 250


def test_cmd_backfill_watch_items_writes_candidate_and_post_exit_rows(monkeypatch) -> None:
    settings = load_settings()
    upserts: list[dict] = []

    class _Repo:
        dataset_fqn = "proj.ds"

        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def list_runtime_tenants(self, *, limit: int = 2000) -> list[str]:
            _ = limit
            return ["tenant-a"]

        def candidate_memory_events_for_structured_backfill(
            self,
            *,
            agent_id: str,
            trading_mode: str = "paper",
            tenant_id: str | None = None,
            limit: int = 1000,
            include_existing: bool = False,
        ) -> list[dict]:
            _ = (agent_id, trading_mode, tenant_id, limit, include_existing)
            return [
                {
                    "event_id": "evt_candidate",
                    "created_at": datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 3, 15, 8, 5, tzinfo=timezone.utc),
                    "agent_id": "gpt",
                    "event_type": "candidate_watchlist",
                    "summary": "AAPL candidate_watchlist: surfaced by recommend_opportunities.",
                    "cycle_id": "cycle_1",
                    "llm_call_id": "llm_1",
                    "payload_json": {
                        "ticker": "AAPL",
                        "source_tools": ["recommend_opportunities"],
                        "candidate_status": "candidate",
                    },
                }
            ]

        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            _ = params
            if "thesis_invalidated" in sql:
                return [
                    {
                        "event_id": "evt_exit",
                        "created_at": datetime(2026, 3, 16, 8, 0, tzinfo=timezone.utc),
                        "updated_at": datetime(2026, 3, 16, 8, 5, tzinfo=timezone.utc),
                        "agent_id": "gpt",
                        "event_type": "thesis_invalidated",
                        "summary": "AAPL thesis invalidated after a sharp drop.",
                        "cycle_id": "cycle_2",
                        "llm_call_id": "llm_2",
                        "payload_json": {
                            "intent": {"ticker": "AAPL", "time_horizon_days": 20},
                            "thesis_id": "thesis_1",
                            "position_action": "exit",
                        },
                    }
                ]
            return []

        def watch_item_by_key(self, *, watch_key: str, tenant_id: str | None = None):
            _ = (watch_key, tenant_id)
            return None

        def upsert_watch_item(self, row: dict, *, tenant_id: str | None = None) -> None:
            upserts.append({"row": dict(row), "tenant_id": tenant_id})

        def append_runtime_audit_log(self, **kwargs):
            _ = kwargs
            return None

    repo = _Repo()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    cli.cmd_backfill_watch_items(
        tenant_ids=["tenant-a"],
        agent_ids=["gpt"],
        live=False,
        dry_run=False,
        include_existing=False,
        limit_per_agent=100,
    )

    assert len(upserts) == 2
    assert upserts[0]["row"]["watch_kind"] == "candidate"
    assert upserts[0]["row"]["ticker"] == "AAPL"
    assert upserts[1]["row"]["watch_kind"] == "post_exit"
    assert upserts[1]["row"]["watch_status"] == "resolved"
