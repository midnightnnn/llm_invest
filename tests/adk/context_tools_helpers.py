from __future__ import annotations

import json

import pandas as pd


class _RepoForPortfolioDiagnosis:
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        _ = lookback_days, sources
        base = {
            "AAPL": [100.0, 101.0, 103.0, 104.0, 106.0, 108.0, 109.0, 111.0, 112.0, 114.0, 116.0, 118.0],
            "MSFT": [200.0, 199.0, 198.0, 201.0, 202.0, 204.0, 205.0, 207.0, 209.0, 210.0, 212.0, 214.0],
            "QQQ": [300.0, 301.0, 302.0, 304.0, 306.0, 307.0, 309.0, 311.0, 312.0, 314.0, 316.0, 318.0],
        }
        return {ticker: base.get(ticker, []) for ticker in tickers}

    def get_daily_close_frame(self, *, tickers, start, end, sources=None, price_field="close_price_krw"):
        _ = (sources, price_field)
        series = {
            "QQQ": [
                ("2026-01-02", 300.0),
                ("2026-01-03", 310.0),
                ("2026-01-04", 315.0),
            ],
        }
        frame = pd.DataFrame(
            {
                token: [px for _, px in series.get(token, [])]
                for token in tickers
                if token in series
            },
            index=pd.to_datetime([ts for ts, _ in series.get(next(iter(tickers), ""), [])]),
        )
        if frame.empty:
            return frame
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        return frame.loc[mask]


class _RepoForPortfolioDiagnosisExact(_RepoForPortfolioDiagnosis):
    def __init__(self) -> None:
        self.frame_calls: list[dict[str, object]] = []

    def get_daily_close_frame(self, *, tickers, start, end, sources=None):  # noqa: ANN001
        self.frame_calls.append(
            {
                "tickers": list(tickers),
                "start": start,
                "end": end,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        frame = pd.DataFrame(
            {"QQQ": [90.0, 100.0, 100.0]},
            index=pd.to_datetime(["2026-01-02", "2026-03-03", "2026-03-27"]),
        )
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        return frame.loc[mask]


class _RepoForPortfolioDiagnosisRaises(_RepoForPortfolioDiagnosis):
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        if int(lookback_days) <= 10:
            raise RuntimeError("no closes")
        return super().get_daily_closes(tickers, lookback_days, sources=sources)


class _RepoForPeerLessons:
    def memory_events_by_ids_any_agent(self, *, event_ids, trading_mode="paper", tenant_id=None):
        _ = (trading_mode, tenant_id)
        rows = {
            "mem_peer": {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "payload_json": json.dumps({"source": "thesis_chain_compaction"}),
            },
            "mem_manual": {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "payload_json": json.dumps({"source": "manual_note"}),
            },
        }
        return [rows[eid] for eid in event_ids if eid in rows]


class _VectorStoreForPeerLessons:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_peer_lessons(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "summary": "Trim single-name exposure after fast gains.",
                "created_date": "2026-03-07",
            },
            {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "summary": "Manual reflection that should be filtered out.",
                "created_date": "2026-03-06",
            },
        ]


class _RepoForResearchBriefingFallback:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        self.calls.append(
            {
                "tickers": list(tickers) if tickers else None,
                "categories": list(categories) if categories else None,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        tenant = str(tenant_id or "").strip().lower()
        if tenant == "tenant-a":
            return []
        if tenant == "midnightnnn":
            rows = [
                {
                    "briefing_id": "pub_global",
                    "category": "global_market",
                    "ticker": "GLOBAL",
                    "headline": "Global",
                    "summary": "global summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_geo",
                    "category": "geopolitical",
                    "ticker": "GEOPOLITICAL",
                    "headline": "Geo",
                    "summary": "geo summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_sector",
                    "category": "sector_trends",
                    "ticker": "SECTOR",
                    "headline": "Sector",
                    "summary": "sector summary",
                    "sources": "[]",
                },
            ]
            if categories:
                allowed = {str(token).strip().lower() for token in categories if str(token).strip()}
                rows = [row for row in rows if str(row.get("category") or "").strip().lower() in allowed]
            return rows[:limit]
        return []
