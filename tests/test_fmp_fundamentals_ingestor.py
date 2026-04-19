from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from arena.open_trading.fmp_fundamentals_ingestor import (
    FMPFundamentalsIngestor,
    _redact_url,
)


def test_fmp_ingestor_uses_stable_statement_endpoints() -> None:
    seen: list[str] = []

    def fake_http(url: str, **_: object) -> list[dict]:
        seen.append(url)
        return []

    ingestor = FMPFundamentalsIngestor(
        api_key="secret-key",
        repo=object(),
        period="quarter",
        limit=2,
        http_fn=fake_http,
    )

    ingestor._fetch_bundle("AAPL")

    assert [urlsplit(url).path for url in seen] == [
        "/stable/income-statement",
        "/stable/balance-sheet-statement",
        "/stable/cash-flow-statement",
    ]
    for url in seen:
        query = parse_qs(urlsplit(url).query)
        assert query["symbol"] == ["AAPL"]
        assert query["period"] == ["quarter"]
        assert query["limit"] == ["4"]
        assert query["apikey"] == ["secret-key"]


def test_fmp_url_redaction_hides_api_key() -> None:
    redacted = _redact_url(
        "https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey=secret-key"
    )

    assert "secret-key" not in redacted
    assert "apikey=%3Credacted%3E" in redacted
