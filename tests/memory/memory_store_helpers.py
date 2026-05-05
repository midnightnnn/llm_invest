from __future__ import annotations


class _FakeVectorStore:
    db = None
    def __init__(self) -> None:
        self.saved = []
    def save_memory_vector(self, **kwargs) -> None:
        self.saved.append(kwargs)
    def search_similar_memories(self, **kwargs) -> list:
        return []
    def search_peer_lessons(self, **kwargs) -> list:
        return []


class _FakeRepo:
    def __init__(self) -> None:
        self.events = []
        self.event_updates = []
        self.score_updates = {}
        self.buy_memories = []
        self.trade_memory_by_order_id = {}
        self.recent_rows = []
        self.active_thesis_rows: dict[str, dict] = {}

    def write_memory_event(self, event) -> None:
        self.events.append(event)
        self.recent_rows.insert(
            0,
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "summary": event.summary,
                "created_at": event.created_at,
            },
        )

    def find_trade_execution_memory_event(self, *, agent_id: str, order_id: str, trading_mode: str = "paper"):
        _ = (agent_id, trading_mode)
        return self.trade_memory_by_order_id.get(order_id)

    def update_memory_event(
        self,
        *,
        event_id: str,
        summary: str,
        payload: dict,
        score: float,
        importance_score: float | None = None,
        outcome_score: float | None = None,
        memory_tier: str | None = None,
        expires_at=None,
        context_tags: dict | None = None,
        primary_regime: str | None = None,
        primary_strategy_tag: str | None = None,
        primary_sector: str | None = None,
        graph_node_id: str | None = None,
        causal_chain_id: str | None = None,
    ) -> None:
        self.event_updates.append(
            {
                "event_id": event_id,
                "summary": summary,
                "payload": payload,
                "score": score,
                "importance_score": importance_score,
                "outcome_score": outcome_score,
                "memory_tier": memory_tier,
                "expires_at": expires_at,
                "context_tags": context_tags,
                "primary_regime": primary_regime,
                "primary_strategy_tag": primary_strategy_tag,
                "primary_sector": primary_sector,
                "graph_node_id": graph_node_id,
                "causal_chain_id": causal_chain_id,
            }
        )

    def recent_memory_events(self, agent_id: str, limit: int, trading_mode: str = "paper") -> list[dict]:
        _ = (agent_id, limit, trading_mode)
        return list(self.recent_rows[:limit])

    def find_buy_memories_for_ticker(self, agent_id: str, ticker: str, limit: int = 5, trading_mode: str = "paper") -> list[dict]:
        return self.buy_memories

    def update_memory_score(self, event_id: str, new_score: float) -> None:
        self.score_updates[event_id] = new_score

    def active_thesis_events(self, *, agent_id: str, tickers: list[str], trading_mode: str = "paper"):
        _ = (agent_id, trading_mode)
        return [self.active_thesis_rows[ticker] for ticker in tickers if ticker in self.active_thesis_rows]
