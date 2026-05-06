from __future__ import annotations

from dataclasses import dataclass


ASSET_BENCHMARK_SOURCE = "asset_benchmark"


@dataclass(frozen=True)
class AssetBenchmark:
    ticker: str
    asset_class: str
    display_name: str
    quote_excd: str = ""


US_ASSET_BENCHMARKS: tuple[AssetBenchmark, ...] = (
    AssetBenchmark("GLD", "gold", "SPDR Gold Shares", "NAS"),
    AssetBenchmark("SLV", "silver", "iShares Silver Trust", "NAS"),
    AssetBenchmark("USO", "oil_energy", "United States Oil Fund", "NAS"),
    AssetBenchmark("TLT", "long_treasury", "iShares 20+ Year Treasury Bond ETF", "NAS"),
    AssetBenchmark("UUP", "usd_currency", "Invesco DB US Dollar Index Bullish Fund", "NAS"),
)


KOSPI_ASSET_BENCHMARKS: tuple[AssetBenchmark, ...] = (
    AssetBenchmark("132030", "gold", "KODEX Gold Futures(H)", "KRX"),
    AssetBenchmark("144600", "silver", "KODEX Silver Futures(H)", "KRX"),
    AssetBenchmark("261220", "oil_energy", "KODEX WTI Oil Futures(H)", "KRX"),
    AssetBenchmark("304660", "long_treasury", "KODEX US 30Y Treasury Futures(H)", "KRX"),
    AssetBenchmark("261240", "usd_currency", "KODEX USD Futures", "KRX"),
)
