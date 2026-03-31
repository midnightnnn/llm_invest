from .client import KISAPIError, OpenTradingClient

__all__ = [
    "KISAPIError",
    "OpenTradingClient",
    "MarketDataSyncService",
    "AccountSyncService",
]


def __getattr__(name: str):
    if name in {"MarketDataSyncService", "AccountSyncService"}:
        from .sync import AccountSyncService, MarketDataSyncService

        if name == "MarketDataSyncService":
            return MarketDataSyncService
        return AccountSyncService
    raise AttributeError(name)
