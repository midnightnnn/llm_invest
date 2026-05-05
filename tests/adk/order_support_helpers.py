from __future__ import annotations

class _RepoForMarketLookup:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def latest_market_features(self, *, tickers, limit, sources=None):
        self.calls.append(
            {
                "tickers": list(tickers),
                "limit": limit,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        return list(self.rows)


class _RepoForAdkGenerate:
    def latest_market_features(self, tickers, limit, sources=None):
        _ = (tickers, limit, sources)
        return []
