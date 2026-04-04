from __future__ import annotations

import json
import types
from datetime import datetime, timedelta, timezone

import pytest

from arena.config import Settings
from arena.open_trading.client import KISAPIError, OpenTradingClient


def _settings(*, suffix: str = "", account_no: str = "") -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=2_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="paper",
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no=account_no,
        kis_account_product_code="01",
        kis_account_key_suffix=suffix,
        kis_env="real",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=350_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=25,
        default_universe=["AAPL"],
        allow_live_trading=False,
    )


def _install_fake_secretmanager(monkeypatch, payload: dict) -> None:
    class _FakeSecretManagerClient:
        def access_secret_version(self, name):
            class _Resp:
                class payload:  # noqa: N801
                    data = json.dumps(payload).encode("utf-8")

            return _Resp()

    fake_module = types.SimpleNamespace(SecretManagerServiceClient=_FakeSecretManagerClient)

    import google.cloud

    monkeypatch.setattr(google.cloud, "secretmanager", fake_module, raising=False)


def test_secret_accounts_selects_by_key_suffix(monkeypatch) -> None:
    payload = {
        "ACCOUNTS": [
            {
                "key_suffix": "MO",
                "app_key": "K1",
                "app_secret": "S1",
                "cano": "11112222",
                "prdt_cd": "01",
            },
            {
                "key_suffix": "CO",
                "app_key": "K2",
                "app_secret": "S2",
                "cano": "33334444",
                "prdt_cd": "02",
            },
        ]
    }
    _install_fake_secretmanager(monkeypatch, payload)
    client = OpenTradingClient(_settings(suffix="CO"))

    app_key, app_secret = client._credentials()
    cano, prdt_cd = client._split_account()

    assert app_key == "K2"
    assert app_secret == "S2"
    assert cano == "33334444"
    assert prdt_cd == "02"


def test_secret_accounts_selects_by_account_number(monkeypatch) -> None:
    payload = {
        "ACCOUNTS": [
            {
                "key_suffix": "MO",
                "app_key": "K1",
                "app_secret": "S1",
                "cano": "11112222",
                "prdt_cd": "01",
            },
            {
                "key_suffix": "CO",
                "app_key": "K2",
                "app_secret": "S2",
                "cano": "33334444",
                "prdt_cd": "02",
            },
        ]
    }
    _install_fake_secretmanager(monkeypatch, payload)
    client = OpenTradingClient(_settings(account_no="1111222201"))

    app_key, app_secret = client._credentials()
    cano, prdt_cd = client._split_account()

    assert app_key == "K1"
    assert app_secret == "S1"
    assert cano == "11112222"
    assert prdt_cd == "01"


def test_demo_credentials_require_paper_keys(monkeypatch) -> None:
    payload = {
        "app_key": "REAL_K",
        "app_secret": "REAL_S",
    }
    _install_fake_secretmanager(monkeypatch, payload)
    settings = _settings()
    settings.kis_env = "demo"
    client = OpenTradingClient(settings)

    with pytest.raises(KISAPIError, match="demo"):
        client._credentials()


def test_get_overseas_price_detail_returns_output(monkeypatch) -> None:
    client = OpenTradingClient(_settings())

    def _fake_request(*, method, path, tr_id, params=None, tr_cont="", retry_on_401=True):
        _ = (method, tr_id, params, tr_cont, retry_on_401)
        assert path == "/uapi/overseas-price/v1/quotations/price-detail"
        return {"rt_cd": "0", "output": {"perx": "21.3", "pbrx": "4.2", "epsx": "10.1"}}, {}

    monkeypatch.setattr(client, "_request", _fake_request)
    out = client.get_overseas_price_detail("AAPL", excd="NAS")
    assert out["perx"] == "21.3"
    assert out["pbrx"] == "4.2"


def test_get_usd_krw_daily_chart_uses_daily_chart_api(monkeypatch) -> None:
    client = OpenTradingClient(_settings())

    def _fake_request(*, method, path, tr_id, params=None, tr_cont="", retry_on_401=True):
        _ = (method, tr_id, tr_cont, retry_on_401)
        assert path == "/uapi/overseas-price/v1/quotations/inquire-daily-chartprice"
        assert params is not None
        assert params["FID_COND_MRKT_DIV_CODE"] == "X"
        assert params["FID_INPUT_ISCD"] == "USDKRW"
        return {"rt_cd": "0", "output2": [{"stck_bsop_date": "20260102", "ovrs_nmix_prpr": "1310.5"}]}, {"tr_cont": ""}

    monkeypatch.setattr(client, "_request", _fake_request)
    rows = client.get_usd_krw_daily_chart(
        symbol="USDKRW",
        start_date="20260101",
        end_date="20260131",
    )
    assert len(rows) == 1
    assert rows[0]["ovrs_nmix_prpr"] == "1310.5"


def test_get_domestic_orderable_cash_returns_nrcvb_buy_amt(monkeypatch) -> None:
    client = OpenTradingClient(_settings(account_no="1234567801"))

    def _fake_request(*, method, path, tr_id, params=None, tr_cont="", retry_on_401=True):
        _ = (method, tr_id, params, tr_cont, retry_on_401)
        assert path == "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        return {
            "rt_cd": "0",
            "output": {
                "nrcvb_buy_amt": "500000",
                "ord_psbl_cash": "490000",
                "ord_psbl_sbst": "0",
                "ruse_psbl_amt": "0",
            },
        }, {}

    monkeypatch.setattr(client, "_request", _fake_request)

    assert client.get_domestic_orderable_cash() == pytest.approx(500000.0)


def test_get_domestic_daily_price_pages_backward_by_end_date(monkeypatch) -> None:
    client = OpenTradingClient(_settings())

    def _row(day: str) -> dict[str, str]:
        return {"stck_bsop_date": day, "stck_clpr": "70000"}

    page1 = []
    cur = datetime(2026, 3, 6, tzinfo=timezone.utc)
    for _ in range(100):
        page1.append(_row(cur.strftime("%Y%m%d")))
        cur -= timedelta(days=1)

    page2 = []
    cur = datetime(2025, 11, 26, tzinfo=timezone.utc)
    for _ in range(70):
        page2.append(_row(cur.strftime("%Y%m%d")))
        cur -= timedelta(days=1)

    seen_end_dates: list[str] = []

    def _fake_request(*, method, path, tr_id, params=None, tr_cont="", retry_on_401=True):
        _ = (method, tr_id, tr_cont, retry_on_401)
        assert path == "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        assert params is not None
        seen_end_dates.append(str(params["FID_INPUT_DATE_2"]))
        if params["FID_INPUT_DATE_2"] == "20260306":
            return {"rt_cd": "0", "output2": page1}, {}
        if params["FID_INPUT_DATE_2"] == "20251126":
            return {"rt_cd": "0", "output2": page2}, {}
        return {"rt_cd": "0", "output2": []}, {}

    monkeypatch.setattr(client, "_request", _fake_request)
    rows = client.get_domestic_daily_price(
        ticker="005930",
        start_date="20251001",
        end_date="20260306",
        max_pages=5,
    )

    assert len(rows) == 157
    assert rows[0]["stck_bsop_date"] == "20260306"
    assert rows[-1]["stck_bsop_date"] == "20251001"
    assert seen_end_dates == ["20260306", "20251126"]


def test_search_overseas_stocks_returns_rows(monkeypatch) -> None:
    client = OpenTradingClient(_settings())

    def _fake_request(*, method, path, tr_id, params=None, tr_cont="", retry_on_401=True):
        _ = (method, tr_id, tr_cont, retry_on_401)
        assert path == "/uapi/overseas-price/v1/quotations/inquire-search"
        assert params is not None
        assert params.get("EXCD") == "NAS"
        return {
            "rt_cd": "0",
            "output2": [
                {"symb": "AAPL", "per": "31.1", "eps": "6.2"},
                {"symb": "MSFT", "per": "34.0", "eps": "12.3"},
            ],
        }, {"tr_cont": ""}

    monkeypatch.setattr(client, "_request", _fake_request)
    rows = client.search_overseas_stocks(excd="NAS", per_min=10.0, per_max=40.0, max_pages=1)
    assert len(rows) == 2
    assert rows[0]["symb"] == "AAPL"


def test_request_refreshes_on_token_expired_msg_cd(monkeypatch) -> None:
    client = OpenTradingClient(_settings())
    client._access_token = "old-token"
    client._token_expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)

    class _Resp:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    calls: list[int] = []

    def _fake_request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _Resp(500, {"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "기간이 만료된 token 입니다."})
        return _Resp(200, {"rt_cd": "0", "output": {"ok": True}})

    refreshed_forces: list[bool] = []

    def _fake_authenticate(force: bool = False):
        refreshed_forces.append(bool(force))
        if force:
            client._access_token = "new-token"
            client._token_expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)

    monkeypatch.setattr(client.session, "request", _fake_request)
    monkeypatch.setattr(client, "_authenticate", _fake_authenticate)
    monkeypatch.setattr(client, "_credentials", lambda: ("k", "s"))

    body, _headers = client._request(
        method="GET",
        path="/uapi/overseas-price/v1/quotations/price",
        tr_id="HHDFS00000300",
        params={"EXCD": "NAS", "SYMB": "AAPL"},
    )
    assert body.get("rt_cd") == "0"
    assert len(calls) == 2
    assert refreshed_forces == [False, True]
