from __future__ import annotations

from arena.open_trading.sec_fundamentals_ingestor import SECFundamentalsIngestor


class _Repo:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.runs: list[dict] = []

    def insert_fundamentals_history_raw(self, rows: list[dict]) -> int:
        self.rows.extend(dict(row) for row in rows)
        return len(rows)

    def append_fundamentals_ingest_run(self, row: dict) -> int:
        self.runs.append(dict(row))
        return 1


def _companyfacts() -> dict:
    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "fy": 2024, "fp": "Q1", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 100.0},
                            {"form": "10-Q", "fy": 2024, "fp": "Q2", "start": "2024-01-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 220.0},
                            {"form": "10-Q", "fy": 2024, "fp": "Q2", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 120.0},
                            {"form": "10-Q", "fy": 2024, "fp": "Q3", "start": "2024-07-01", "end": "2024-09-30", "filed": "2024-11-01", "val": 130.0},
                            {"form": "10-K", "fy": 2024, "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 500.0},
                            # Comparative prior-period duplicate in a later filing; should be ignored.
                            {"form": "10-Q", "fy": 2025, "fp": "Q1", "start": "2024-01-01", "end": "2024-03-31", "filed": "2025-05-01", "val": 100.0},
                            # Bad comparative metadata occasionally reports a fiscal year far beyond period_end.
                            {"form": "10-K", "fy": 2027, "fp": "FY", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 900.0},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "fy": 2024, "fp": "Q1", "start": "2024-01-01", "end": "2024-03-31", "filed": "2024-05-01", "val": 10.0},
                            {"form": "10-Q", "fy": 2024, "fp": "Q2", "start": "2024-04-01", "end": "2024-06-30", "filed": "2024-08-01", "val": 20.0},
                            {"form": "10-Q", "fy": 2024, "fp": "Q3", "start": "2024-07-01", "end": "2024-09-30", "filed": "2024-11-01", "val": 30.0},
                            {"form": "10-K", "fy": 2024, "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 100.0},
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "fy": 2024, "fp": "Q1", "end": "2024-03-31", "filed": "2024-05-01", "val": 1000.0},
                            {"form": "10-K", "fy": 2024, "fp": "FY", "end": "2024-12-31", "filed": "2025-02-20", "val": 1200.0},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "fy": 2024, "fp": "Q1", "end": "2024-03-31", "filed": "2024-05-01", "val": 400.0},
                            {"form": "10-K", "fy": 2024, "fp": "FY", "end": "2024-12-31", "filed": "2025-02-20", "val": 450.0},
                        ]
                    }
                },
            }
        }
    }


def test_sec_companyfacts_extracts_quarter_rows_and_derives_q4() -> None:
    def http_json(url: str, **_: object) -> dict:
        if url.endswith("company_tickers.json"):
            return {"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}}
        return _companyfacts()

    ingestor = SECFundamentalsIngestor(
        repo=_Repo(),
        user_agent="LLM Arena tests@example.com",
        http_json=http_json,
        sleep_seconds=0.0,
    )

    rows = ingestor.records_for_ticker("AAPL")

    by_period = {(row["fiscal_year"], row["fiscal_quarter"]): row for row in rows}
    assert sorted(by_period) == [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
    assert by_period[(2024, 2)]["revenue"] == 120.0
    assert by_period[(2024, 4)]["revenue"] == 150.0
    assert by_period[(2024, 4)]["net_income"] == 40.0
    assert all(row["source"] == "sec_companyfacts" for row in rows)
    assert all(row["announcement_date_source"] == "sec_filed" for row in rows)


def test_sec_run_batches_history_rows_and_appends_metadata() -> None:
    repo = _Repo()

    def http_json(url: str, **_: object) -> dict:
        if url.endswith("company_tickers.json"):
            return {"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}}
        return _companyfacts()

    ingestor = SECFundamentalsIngestor(
        repo=repo,
        user_agent="LLM Arena tests@example.com",
        http_json=http_json,
        sleep_seconds=0.0,
        write_batch_size=2,
    )

    result = ingestor.run(tickers=["AAPL", "MISSING"])

    assert result.status == "ok"
    assert result.tickers_attempted == 2
    assert result.tickers_succeeded == 1
    assert result.quarters_inserted == 4
    assert len(repo.rows) == 4
    assert repo.runs[0]["source"] == "sec_companyfacts"
