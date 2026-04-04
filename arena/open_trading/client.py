from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from arena.config import Settings
from arena.open_trading.exchange_codes import normalize_us_order_exchange, target_market_default_us_order_exchange

logger = logging.getLogger(__name__)

PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"
PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"


def _to_float(value: object, default: float = 0.0) -> float:
    """Converts mixed API values to float with safe fallback."""
    try:
        if value is None:
            return default
        text = str(value).strip().replace(",", "")
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


class KISAPIError(RuntimeError):
    """Represents a failed Korea Investment Open API call."""


class OpenTradingClient:
    """Thin REST client for open-trading-api endpoints used by the arena."""

    _TOKEN_CACHE: dict[tuple[str, str], tuple[str, datetime]] = {}
    _TOKEN_LOCK = threading.Lock()
    _TOKEN_EXPIRED_CODES: set[str] = {"EGW00123"}

    def __init__(self, settings: Settings, timeout_seconds: int | None = None):
        self.settings = settings
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else max(1, settings.kis_http_timeout_seconds)
        )
        self.session = requests.Session()
        self._access_token = ""
        self._token_expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        self._secret_loaded = False
        self._secret_payload: dict[str, Any] = {}
        self._firestore_token_cache = None

    @property
    def is_demo(self) -> bool:
        """Returns True when the client uses mock trading credentials."""
        return self.settings.kis_env == "demo"

    @property
    def base_url(self) -> str:
        """Returns the REST base URL for selected live/mock environment."""
        return PAPER_BASE_URL if self.is_demo else PROD_BASE_URL

    def _firestore_cache(self):
        """Returns Firestore cache helper when enabled."""
        backend = (os.getenv("KIS_TOKEN_CACHE_BACKEND", "memory") or "memory").strip().lower()
        if backend != "firestore":
            return None
        if self._firestore_token_cache is not None:
            return self._firestore_token_cache
        try:
            # Guard import explicitly so missing firestore deps do not break trading calls.
            from google.cloud import firestore as _firestore  # noqa: F401
            from arena.open_trading.token_cache import FirestoreTokenCache

            collection = (os.getenv("KIS_TOKEN_CACHE_COLLECTION", "api_tokens") or "api_tokens").strip() or "api_tokens"
            self._firestore_token_cache = FirestoreTokenCache(
                project=self.settings.google_cloud_project,
                collection=collection,
            )
        except Exception as exc:
            logger.warning("[yellow]Firestore token cache disabled[/yellow] err=%s", str(exc))
            self._firestore_token_cache = None
        return self._firestore_token_cache


    def _select_secret_account(self, accounts: list[dict[str, Any]]) -> dict[str, Any]:
        """Selects one account from Secret Manager payload by suffix/account match."""
        if not accounts:
            return {}

        suffix = self.settings.kis_account_key_suffix.strip().upper()
        if suffix:
            for account in accounts:
                if str(account.get("key_suffix", "")).strip().upper() == suffix:
                    return account

        digits = re.sub(r"\D", "", self.settings.kis_account_no)
        if digits:
            for account in accounts:
                cano = re.sub(r"\D", "", str(account.get("cano", "")))
                prdt_cd = re.sub(r"\D", "", str(account.get("prdt_cd", "")))
                if len(digits) >= 10 and digits[:8] == cano and digits[8:10] == prdt_cd:
                    return account
                if len(digits) == 8 and digits == cano:
                    return account

        return accounts[0]

    def _load_secret_payload(self) -> None:
        """Loads KIS credentials from Secret Manager once per runtime."""
        if self._secret_loaded:
            return

        self._secret_loaded = True
        secret_name = self.settings.kis_secret_name.strip()
        if not secret_name:
            return

        project = self.settings.google_cloud_project.strip()
        if not project:
            return

        try:
            from google.cloud import secretmanager

            client = secretmanager.SecretManagerServiceClient()
            resource_name = f"projects/{project}/secrets/{secret_name}/versions/{self.settings.kis_secret_version}"
            response = client.access_secret_version(name=resource_name)
            payload_text = response.payload.data.decode("utf-8")
            data = json.loads(payload_text)
            if isinstance(data, list):
                data = {"ACCOUNTS": data}
            if isinstance(data, dict):
                payload = dict(data)
                raw_accounts = payload.get("ACCOUNTS") or payload.get("accounts")
                if isinstance(raw_accounts, list):
                    accounts = [item for item in raw_accounts if isinstance(item, dict)]
                    if accounts:
                        payload.update(self._select_secret_account(accounts))
                self._secret_payload = payload
                suffix = str(payload.get("key_suffix", "")).strip() or "-"
                logger.info(
                    "[cyan]KIS secret loaded[/cyan] source=secret-manager name=%s key_suffix=%s",
                    secret_name,
                    suffix,
                )
        except Exception as exc:
            logger.warning("[yellow]KIS secret load skipped[/yellow] reason=%s", str(exc))

    def _credentials(self) -> tuple[str, str]:
        """Returns active app key/secret pair for current environment."""
        self._load_secret_payload()
        secret = self._secret_payload

        if self.is_demo:
            app_key = self.settings.kis_paper_api_key or str(secret.get("paper_app_key") or secret.get("paper_appkey") or "")
            app_secret = self.settings.kis_paper_api_secret or str(secret.get("paper_app_secret") or secret.get("paper_appsecret") or "")
        else:
            app_key = self.settings.kis_api_key or str(secret.get("app_key") or secret.get("appkey") or "")
            app_secret = self.settings.kis_api_secret or str(secret.get("app_secret") or secret.get("appsecret") or "")

        app_key = app_key.strip()
        app_secret = app_secret.strip()

        if not app_key or not app_secret:
            mode = "demo" if self.is_demo else "real"
            raise KISAPIError(f"KIS credentials are missing for {mode} environment")

        return app_key, app_secret

    def _split_account(self) -> tuple[str, str]:
        """Splits configured account into front(8) and product(2)."""
        self._load_secret_payload()
        secret = self._secret_payload

        account_no = self.settings.kis_account_no.strip()
        if not account_no:
            cano = str(secret.get("cano", "")).strip()
            prdt_cd = str(secret.get("prdt_cd", "") or self.settings.kis_account_product_code).strip()
            if cano and prdt_cd:
                account_no = f"{cano}{prdt_cd}"

        digits = re.sub(r"\D", "", account_no)
        if len(digits) >= 10:
            return digits[:8], digits[8:10]

        if len(digits) == 8:
            return digits, self.settings.kis_account_product_code or str(secret.get("prdt_cd", "01"))

        raise KISAPIError("KIS account is missing. set KIS_ACCOUNT_NO or Secret Manager cano/prdt_cd")

    def _to_tr_id(self, tr_id: str) -> str:
        """Converts prod TR-ID to mock TR-ID when needed."""
        if self.is_demo and tr_id and tr_id[0] in {"T", "J", "C"}:
            return f"V{tr_id[1:]}"
        return tr_id

    def _authenticate(self, force: bool = False) -> None:
        """Issues or refreshes access token when missing or near expiry."""
        now = datetime.now(timezone.utc)
        if not force and self._access_token and now < self._token_expires_at - timedelta(minutes=3):
            return

        app_key, app_secret = self._credentials()
        cache_key = (self.base_url, app_key)
        with self._TOKEN_LOCK:
            cached = self._TOKEN_CACHE.get(cache_key)
        if not force and cached:
            token, expires_at = cached
            if token and now < expires_at - timedelta(minutes=3):
                self._access_token = token
                self._token_expires_at = expires_at
                return

        fs_cache = self._firestore_cache()
        if not force and fs_cache:
            try:
                rec = fs_cache.get(base_url=self.base_url, app_key=app_key)
            except Exception as exc:
                logger.warning("[yellow]Token cache read skipped[/yellow] err=%s", str(exc))
                rec = None
            if rec and now < rec.expires_at - timedelta(minutes=3):
                self._access_token = rec.token
                self._token_expires_at = rec.expires_at
                with self._TOKEN_LOCK:
                    self._TOKEN_CACHE[cache_key] = (rec.token, rec.expires_at)
                return

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret,
        }
        headers = {"content-type": "application/json"}

        max_retries = max(0, int(self.settings.kis_http_max_retries))
        base = max(self.settings.kis_http_backoff_base_seconds, 0.2)
        cap = max(self.settings.kis_http_backoff_max_seconds, base)

        def backoff(attempt: int, *, rate_limited: bool) -> float:
            raw = base * (2**attempt)
            raw = min(cap, raw)
            if rate_limited:
                raw = max(raw, 61.0)
            return raw + (random.random() * min(0.25 * base, 0.5))

        last_status = 0
        last_body = ""
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt >= max_retries:
                    raise KISAPIError(f"KIS auth request failed: {exc}") from exc
                sleep = backoff(attempt, rate_limited=False)
                logger.warning(
                    "[yellow]KIS auth retrying[/yellow] attempt=%d sleep=%.2fs reason=network",
                    attempt + 1,
                    sleep,
                )
                time.sleep(sleep)
                continue

            if response.status_code == 200:
                body = response.json()
                token = str(body.get("access_token", "")).strip()
                if not token:
                    raise KISAPIError(f"KIS auth returned empty token: {body}")

                expires_text = str(body.get("access_token_token_expired", "")).strip()
                if expires_text:
                    try:
                        expires_at = datetime.strptime(expires_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        expires_at = now + timedelta(hours=23)
                else:
                    expires_at = now + timedelta(hours=23)

                self._access_token = token
                self._token_expires_at = expires_at
                with self._TOKEN_LOCK:
                    self._TOKEN_CACHE[cache_key] = (token, expires_at)
                if fs_cache:
                    try:
                        from arena.open_trading.token_cache import TokenRecord

                        fs_cache.set(base_url=self.base_url, app_key=app_key, record=TokenRecord(token=token, expires_at=expires_at))
                    except Exception as exc:
                        logger.warning("[yellow]Token cache write skipped[/yellow] err=%s", str(exc))
                return

            last_status = response.status_code
            last_body = response.text[:300]

            if fs_cache:
                try:
                    rec = fs_cache.get(base_url=self.base_url, app_key=app_key)
                except Exception as exc:
                    logger.warning("[yellow]Token cache read skipped[/yellow] err=%s", str(exc))
                    rec = None
                if rec and now < rec.expires_at - timedelta(minutes=1):
                    self._access_token = rec.token
                    self._token_expires_at = rec.expires_at
                    with self._TOKEN_LOCK:
                        self._TOKEN_CACHE[cache_key] = (rec.token, rec.expires_at)
                    return

            # KIS imposes strict token issuance limits; fall back to cached token when possible.
            if cached:
                token, expires_at = cached
                if token and now < expires_at - timedelta(minutes=1):
                    self._access_token = token
                    self._token_expires_at = expires_at
                    return

            err_code = ""
            try:
                body = response.json()
                err_code = str(body.get("error_code", "")).strip()
            except ValueError:
                body = {}

            if attempt < max_retries:
                rate_limited = response.status_code in {403, 429} and err_code == "EGW00133"
                retryable = response.status_code in {403, 429, 500, 502, 503, 504}
                if retryable:
                    sleep = backoff(attempt, rate_limited=rate_limited)
                    logger.warning(
                        "[yellow]KIS auth retrying[/yellow] attempt=%d sleep=%.2fs status=%d code=%s",
                        attempt + 1,
                        sleep,
                        response.status_code,
                        err_code or "-",
                    )
                    time.sleep(sleep)
                    continue

            raise KISAPIError(f"KIS auth failed status={last_status} body={last_body}")

        raise KISAPIError(f"KIS auth failed status={last_status} body={last_body}")


    def _request(
        self,
        *,
        method: str,
        path: str,
        tr_id: str,
        params: dict[str, Any] | None = None,
        tr_cont: str = "",
        retry_on_401: bool = True,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Sends one authenticated API call with retries for transient failures."""
        url = f"{self.base_url}{path}"
        verb = method.upper()
        retryable_statuses = {429, 500, 502, 503, 504}
        max_retries = max(0, int(self.settings.kis_http_max_retries))

        def backoff_seconds(attempt: int, retry_after: str = "") -> float:
            base = max(self.settings.kis_http_backoff_base_seconds, 0.01)
            cap = max(self.settings.kis_http_backoff_max_seconds, base)
            sleep = min(cap, base * (2**attempt))
            if retry_after.strip().isdigit():
                sleep = max(sleep, float(retry_after.strip()))
            jitter = random.random() * min(0.25 * base, 0.5)
            return sleep + jitter

        attempt = 0
        refreshed = False

        while True:
            self._authenticate(force=refreshed)
            app_key, app_secret = self._credentials()

            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {self._access_token}",
                "appkey": app_key,
                "appsecret": app_secret,
                "tr_id": self._to_tr_id(tr_id),
                "custtype": "P",
                "tr_cont": tr_cont,
            }

            request_kwargs: dict[str, Any] = {
                "headers": headers,
                "timeout": self.timeout_seconds,
            }
            if verb == "GET":
                request_kwargs["params"] = params or {}
            else:
                request_kwargs["data"] = json.dumps(params or {})

            try:
                response = self.session.request(method=verb, url=url, **request_kwargs)
            except requests.RequestException as exc:
                if attempt >= max_retries:
                    raise KISAPIError(f"KIS request failed path={path} err={exc}") from exc
                sleep = backoff_seconds(attempt)
                logger.warning(
                    "[yellow]KIS request retrying[/yellow] path=%s attempt=%d sleep=%.2fs reason=network",
                    path,
                    attempt + 1,
                    sleep,
                )
                time.sleep(sleep)
                attempt += 1
                continue

            token_expired = False
            token_msg_cd = ""
            token_msg = ""
            try:
                body_preview = response.json()
                token_msg_cd = str(body_preview.get("msg_cd", "")).strip()
                token_msg = str(body_preview.get("msg1", "")).strip()
                token_expired = token_msg_cd in self._TOKEN_EXPIRED_CODES
            except ValueError:
                token_expired = False

            if token_expired and retry_on_401 and not refreshed:
                logger.warning(
                    "[yellow]KIS token expired; refreshing[/yellow] path=%s attempt=%d msg_cd=%s msg=%s",
                    path,
                    attempt + 1,
                    token_msg_cd,
                    token_msg or "-",
                )
                refreshed = True
                attempt += 1
                continue

            if response.status_code == 401 and retry_on_401 and not refreshed:
                refreshed = True
                attempt += 1
                continue

            if response.status_code in retryable_statuses and attempt < max_retries:
                retry_after = str(response.headers.get("retry-after", ""))
                sleep = backoff_seconds(attempt, retry_after)
                logger.warning(
                    "[yellow]KIS request retrying[/yellow] path=%s attempt=%d sleep=%.2fs status=%d",
                    path,
                    attempt + 1,
                    sleep,
                    response.status_code,
                )
                time.sleep(sleep)
                attempt += 1
                continue

            if response.status_code != 200:
                raise KISAPIError(
                    f"KIS request failed status={response.status_code} path={path} body={response.text[:300]}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                if attempt >= max_retries:
                    raise KISAPIError(f"KIS response JSON decode failed path={path}") from exc
                sleep = backoff_seconds(attempt)
                logger.warning(
                    "[yellow]KIS JSON decode retrying[/yellow] path=%s attempt=%d sleep=%.2fs",
                    path,
                    attempt + 1,
                    sleep,
                )
                time.sleep(sleep)
                attempt += 1
                continue

            rt_cd = str(body.get("rt_cd", "0"))
            if rt_cd != "0":
                msg_cd = body.get("msg_cd", "")
                msg1 = body.get("msg1", "")
                if str(msg_cd).strip() in self._TOKEN_EXPIRED_CODES and retry_on_401 and not refreshed:
                    logger.warning(
                        "[yellow]KIS token expired in body; refreshing[/yellow] path=%s attempt=%d msg_cd=%s msg=%s",
                        path,
                        attempt + 1,
                        str(msg_cd),
                        str(msg1) or "-",
                    )
                    refreshed = True
                    attempt += 1
                    continue
                raise KISAPIError(f"KIS rt_cd={rt_cd} msg_cd={msg_cd} msg={msg1} path={path}")

            return body, {str(k).lower(): str(v) for k, v in response.headers.items()}

    def get_overseas_price(self, ticker: str, excd: str | None = None) -> dict[str, Any]:
        """Returns current overseas quote snapshot for one ticker."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        body, _ = self._request(
            method="GET",
            path="/uapi/overseas-price/v1/quotations/price",
            tr_id="HHDFS00000300",
            params={
                "AUTH": "",
                "EXCD": (excd or self.settings.kis_overseas_quote_excd).strip().upper(),
                "SYMB": symbol,
            },
        )
        output = body.get("output") or {}
        if isinstance(output, list):
            return output[0] if output else {}
        return dict(output)

    def get_overseas_daily_price(
        self,
        ticker: str,
        excd: str | None = None,
        bymd: str = "",
        gubn: str = "0",
        modp: str = "1",
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Returns paginated overseas OHLCV rows for one ticker."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-price/v1/quotations/dailyprice",
                tr_id="HHDFS76240000",
                tr_cont=tr_cont,
                params={
                    "AUTH": "",
                    "EXCD": (excd or self.settings.kis_overseas_quote_excd).strip().upper(),
                    "SYMB": symbol,
                    "GUBN": gubn,
                    "BYMD": bymd,
                    "MODP": modp,
                },
            )

            page_rows = body.get("output2") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows

    def get_overseas_period_rights(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        excd: str | None = None,
        rght_type_cd: str = "03",
        max_pages: int = 4,
    ) -> list[dict[str, Any]]:
        """Returns paginated overseas rights/dividend records for one ticker.

        Uses KIS ``/uapi/overseas-price/v1/quotations/period-rights`` endpoint
        (TR_ID ``CTRGT011R``).  ``rght_type_cd="03"`` restricts to cash
        dividends.  Field names may vary; callers should probe multiple
        candidate keys and log raw responses at DEBUG level.
        """
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for page in range(max_pages):
            try:
                body, headers = self._request(
                    method="GET",
                    path="/uapi/overseas-price/v1/quotations/period-rights",
                    tr_id="CTRGT011R",
                    tr_cont=tr_cont,
                    params={
                        "AUTH": "",
                        "EXCD": (excd or self.settings.kis_overseas_quote_excd).strip().upper(),
                        "SYMB": symbol,
                        "GUBN": rght_type_cd,
                        "BYMD": end_date.replace("-", ""),
                        "STDT": start_date.replace("-", ""),
                    },
                )
            except Exception as exc:
                if page == 0:
                    raise
                logger.warning(
                    "[yellow]period_rights pagination stopped[/yellow] ticker=%s page=%d err=%s",
                    symbol, page, str(exc),
                )
                break

            logger.debug("period_rights raw response ticker=%s page=%d body=%s", symbol, page, body)

            page_rows = body.get("output2") or body.get("output") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows

    def get_overseas_index_daily(
        self,
        symbol: str,
        *,
        excd: str = "NAS",
        period: str = "0",
        start_date: str = "",
        end_date: str = "",
        max_pages: int = 2,
    ) -> list[dict[str, Any]]:
        """해외지수(S&P500, NASDAQ, DJI 등) 기간별 시세를 조회한다."""
        from datetime import datetime, timedelta

        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol is required")

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-price/v1/quotations/inquire-daily-chartprice",
                tr_id="FHKST03030100",
                tr_cont=tr_cont,
                params={
                    "FID_COND_MRKT_DIV_CODE": "N",
                    "FID_INPUT_ISCD": symbol,
                    "FID_INPUT_DATE_1": start_date,
                    "FID_INPUT_DATE_2": end_date,
                    "FID_PERIOD_DIV_CODE": period,
                },
            )

            page_rows = body.get("output2") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows

    def get_domestic_index_daily(
        self,
        iscd: str,
        *,
        start_date: str = "",
        end_date: str = "",
        period: str = "D",
        max_pages: int = 1,
    ) -> list[dict[str, Any]]:
        """국내 업종지수 기간별 시세 (KOSPI=0001, KOSPI200=0028, KOSDAQ=1001)."""
        from datetime import datetime, timedelta

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
                tr_id="FHKUP03500100",
                tr_cont=tr_cont,
                params={
                    "FID_COND_MRKT_DIV_CODE": "U",
                    "FID_INPUT_ISCD": iscd,
                    "FID_INPUT_DATE_1": start_date,
                    "FID_INPUT_DATE_2": end_date,
                    "FID_PERIOD_DIV_CODE": period,
                },
            )
            page_rows = body.get("output2") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))
            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"
        return rows

    def get_overseas_daily_chartprice(
        self,
        *,
        market_div_code: str,
        symbol: str,
        start_date: str = "",
        end_date: str = "",
        period: str = "D",
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Returns paginated daily chart rows for overseas indices/fx/futures APIs."""
        token = str(symbol or "").strip().upper()
        if not token:
            raise ValueError("symbol is required")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-price/v1/quotations/inquire-daily-chartprice",
                tr_id="FHKST03030100",
                tr_cont=tr_cont,
                params={
                    "FID_COND_MRKT_DIV_CODE": str(market_div_code or "N").strip().upper(),
                    "FID_INPUT_ISCD": token,
                    "FID_INPUT_DATE_1": str(start_date or "").strip(),
                    "FID_INPUT_DATE_2": str(end_date or "").strip(),
                    "FID_PERIOD_DIV_CODE": str(period or "D").strip().upper(),
                },
            )

            page_rows = body.get("output2") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows

    def get_usd_krw_daily_chart(
        self,
        *,
        symbol: str,
        start_date: str = "",
        end_date: str = "",
        market_div_code: str = "X",
        period: str = "D",
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Returns USD/KRW daily chart rows using the overseas chart API."""
        return self.get_overseas_daily_chartprice(
            market_div_code=market_div_code,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            max_pages=max_pages,
        )

    def get_overseas_price_detail(self, ticker: str, excd: str | None = None) -> dict[str, Any]:
        """Returns overseas quote details (PER/PBR/EPS/BPS etc.) for one ticker."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")

        body, _ = self._request(
            method="GET",
            path="/uapi/overseas-price/v1/quotations/price-detail",
            tr_id="HHDFS76200200",
            params={
                "AUTH": "",
                "EXCD": (excd or self.settings.kis_overseas_quote_excd).strip().upper(),
                "SYMB": symbol,
            },
        )

        output = body.get("output") or {}
        if isinstance(output, list):
            return output[0] if output else {}
        return dict(output)

    def search_overseas_stocks(
        self,
        *,
        excd: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        per_min: float | None = None,
        per_max: float | None = None,
        eps_min: float | None = None,
        eps_max: float | None = None,
        max_pages: int = 4,
    ) -> list[dict[str, Any]]:
        """Screens overseas stocks by basic conditions (price/PER/EPS)."""

        def _fmt_num(value: float) -> str:
            return f"{float(value):g}"

        def _range_params(start: float | None, end: float | None) -> tuple[str, str, str]:
            if start is None and end is None:
                return "", "", ""
            lo = start if start is not None else end
            hi = end if end is not None else start
            if lo is None or hi is None:
                return "", "", ""
            if float(lo) > float(hi):
                lo, hi = hi, lo
            return "1", _fmt_num(float(lo)), _fmt_num(float(hi))

        if max_pages <= 0:
            raise ValueError("max_pages must be positive")

        co_yn_pricecur, co_st_pricecur, co_en_pricecur = _range_params(price_min, price_max)
        co_yn_per, co_st_per, co_en_per = _range_params(per_min, per_max)
        co_yn_eps, co_st_eps, co_en_eps = _range_params(eps_min, eps_max)

        tr_cont = ""
        keyb = ""
        rows: list[dict[str, Any]] = []
        pages = max(1, min(int(max_pages), 20))

        for _ in range(pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-price/v1/quotations/inquire-search",
                tr_id="HHDFS76410000",
                tr_cont=tr_cont,
                params={
                    "AUTH": "",
                    "EXCD": (excd or self.settings.kis_overseas_quote_excd).strip().upper(),
                    "CO_YN_PRICECUR": co_yn_pricecur,
                    "CO_ST_PRICECUR": co_st_pricecur,
                    "CO_EN_PRICECUR": co_en_pricecur,
                    "CO_YN_RATE": "",
                    "CO_ST_RATE": "",
                    "CO_EN_RATE": "",
                    "CO_YN_VALX": "",
                    "CO_ST_VALX": "",
                    "CO_EN_VALX": "",
                    "CO_YN_SHAR": "",
                    "CO_ST_SHAR": "",
                    "CO_EN_SHAR": "",
                    "CO_YN_VOLUME": "",
                    "CO_ST_VOLUME": "",
                    "CO_EN_VOLUME": "",
                    "CO_YN_AMT": "",
                    "CO_ST_AMT": "",
                    "CO_EN_AMT": "",
                    "CO_YN_EPS": co_yn_eps,
                    "CO_ST_EPS": co_st_eps,
                    "CO_EN_EPS": co_en_eps,
                    "CO_YN_PER": co_yn_per,
                    "CO_ST_PER": co_st_per,
                    "CO_EN_PER": co_en_per,
                    "KEYB": keyb,
                },
            )

            output2 = body.get("output2") or []
            if isinstance(output2, dict):
                output2 = [output2]
            rows.extend(dict(row) for row in output2 if isinstance(row, dict))

            output1 = body.get("output1") or {}
            if isinstance(output1, list):
                output1 = output1[0] if output1 else {}
            next_keyb = str(output1.get("keyb") or body.get("keyb") or "").strip()

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            tr_cont = "N"
            keyb = next_keyb

        return rows

    def place_overseas_order(
        self,
        *,
        ticker: str,
        side: str,
        quantity: int,
        limit_price: float,
        exchange_code: str | None = None,
        ord_dvsn: str = "00",
    ) -> dict[str, Any]:
        """Places one overseas stock limit order and returns API output payload."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if limit_price <= 0:
            raise ValueError("limit_price must be positive")

        side_key = side.strip().lower()
        excd = normalize_us_order_exchange(exchange_code)
        if not excd:
            excd = target_market_default_us_order_exchange(self.settings.kis_target_market)
        if not excd:
            raise ValueError("exchange_code is required for overseas orders")
        buy_tr_ids = {
            "NASD": "TTTT1002U",
            "NYSE": "TTTT1002U",
            "AMEX": "TTTT1002U",
            "SEHK": "TTTS1002U",
            "SHAA": "TTTS0202U",
            "SZAA": "TTTS0305U",
            "TKSE": "TTTS0308U",
            "HASE": "TTTS0311U",
            "VNSE": "TTTS0311U",
        }
        sell_tr_ids = {
            "NASD": "TTTT1006U",
            "NYSE": "TTTT1006U",
            "AMEX": "TTTT1006U",
            "SEHK": "TTTS1001U",
            "SHAA": "TTTS1005U",
            "SZAA": "TTTS0304U",
            "TKSE": "TTTS0307U",
            "HASE": "TTTS0310U",
            "VNSE": "TTTS0310U",
        }

        if side_key == "buy":
            tr_id = buy_tr_ids.get(excd)
            sll_type = ""
        elif side_key == "sell":
            tr_id = sell_tr_ids.get(excd)
            sll_type = "00"
        else:
            raise ValueError("side must be buy or sell")

        if not tr_id:
            raise ValueError(f"unsupported overseas exchange code: {excd}")

        cano, prod = self._split_account()
        body, _ = self._request(
            method="POST",
            path="/uapi/overseas-stock/v1/trading/order",
            tr_id=tr_id,
            params={
                "CANO": cano,
                "ACNT_PRDT_CD": prod,
                "OVRS_EXCG_CD": excd,
                "PDNO": ticker.strip().upper(),
                "ORD_QTY": str(quantity),
                "OVRS_ORD_UNPR": f"{limit_price:.2f}",
                "CTAC_TLNO": "",
                "MGCO_APTM_ODNO": "",
                "SLL_TYPE": sll_type,
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": ord_dvsn,
            },
        )

        output = body.get("output") or {}
        if isinstance(output, list):
            output = output[0] if output else {}
        return {
            "output": dict(output),
            "msg_cd": str(body.get("msg_cd", "")),
            "msg1": str(body.get("msg1", "")),
        }

    def get_domestic_price(self, ticker: str, market_div_code: str = "J") -> dict[str, Any]:
        """Returns current domestic stock quote snapshot for one ticker."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
        )
        output = body.get("output") or {}
        if isinstance(output, list):
            return output[0] if output else {}
        return dict(output)

    def get_domestic_daily_price(
        self,
        ticker: str,
        *,
        start_date: str,
        end_date: str,
        market_div_code: str = "J",
        period_div_code: str = "D",
        org_adj_prc: str = "1",
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Returns domestic daily OHLCV rows for one ticker."""
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")

        try:
            start_dt = datetime.strptime(str(start_date), "%Y%m%d").date()
            end_dt = datetime.strptime(str(end_date), "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError("start_date/end_date must be YYYYMMDD") from exc

        current_end = end_dt
        pages = 0
        rows_by_date: dict[str, dict[str, Any]] = {}

        while current_end >= start_dt and pages < max_pages:
            body, _ = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                tr_id="FHKST03010100",
                params={
                    "FID_COND_MRKT_DIV_CODE": market_div_code,
                    "FID_INPUT_ISCD": symbol,
                    "FID_INPUT_DATE_1": start_date,
                    "FID_INPUT_DATE_2": current_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": period_div_code,
                    "FID_ORG_ADJ_PRC": org_adj_prc,
                },
            )
            rows = body.get("output2") or []
            if isinstance(rows, dict):
                rows = [rows]
            page_rows = [dict(row) for row in rows if isinstance(row, dict)]
            if not page_rows:
                break

            oldest_dt: date | None = None
            for row in page_rows:
                raw_date = str(row.get("stck_bsop_date") or "").strip()
                if len(raw_date) != 8 or not raw_date.isdigit():
                    continue
                try:
                    row_dt = datetime.strptime(raw_date, "%Y%m%d").date()
                except ValueError:
                    continue
                if row_dt < start_dt or row_dt > end_dt:
                    continue
                rows_by_date[raw_date] = row
                if oldest_dt is None or row_dt < oldest_dt:
                    oldest_dt = row_dt

            if oldest_dt is None or oldest_dt <= start_dt:
                break

            next_end = oldest_dt - timedelta(days=1)
            if next_end >= current_end:
                break
            current_end = next_end
            pages += 1

        return [rows_by_date[key] for key in sorted(rows_by_date.keys(), reverse=True)]

    def get_domestic_financial_ratio(
        self,
        ticker: str,
        *,
        market_div_code: str = "J",
        div_cls_code: str = "1",
    ) -> list[dict[str, Any]]:
        """Returns domestic stock financial ratios (EPS, BPS, ROE, debt ratio, etc.).

        div_cls_code: "0" = annual, "1" = quarterly.
        """
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("ticker is required")
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/finance/financial-ratio",
            tr_id="FHKST66430300",
            params={
                "FID_DIV_CLS_CODE": div_cls_code,
                "FID_COND_MRKT_DIV_CODE": market_div_code,
                "FID_INPUT_ISCD": symbol,
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(row) for row in rows if isinstance(row, dict)]

    def get_domestic_market_cap_ranking(
        self,
        *,
        market_scope: str = "0001",
        div_cls_code: str = "0",
    ) -> list[dict[str, Any]]:
        """Returns domestic market-cap ranking rows for KRX equities."""
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/ranking/market-cap",
            tr_id="FHPST01740000",
            params={
                "fid_input_price_2": "",
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20174",
                "fid_div_cls_code": div_cls_code,
                "fid_input_iscd": market_scope,
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_input_price_1": "",
                "fid_vol_cnt": "",
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(row) for row in rows if isinstance(row, dict)]

    def get_domestic_top_interest_stock(
        self,
        *,
        market_scope: str = "0001",
    ) -> list[dict[str, Any]]:
        """Returns domestic top-interest ranking rows for KRX equities."""
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/ranking/top-interest-stock",
            tr_id="FHPST01800000",
            params={
                "fid_input_iscd_2": "000000",
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20180",
                "fid_input_iscd": market_scope,
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_input_price_1": "",
                "fid_input_price_2": "",
                "fid_vol_cnt": "",
                "fid_div_cls_code": "0",
                "fid_input_cnt_1": "1",
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(row) for row in rows if isinstance(row, dict)]

    def get_domestic_volume_rank(
        self,
        *,
        market_scope: str = "0001",
    ) -> list[dict[str, Any]]:
        """Returns domestic volume ranking rows for KRX equities."""
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/volume-rank",
            tr_id="FHPST01710000",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": market_scope,
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(row) for row in rows if isinstance(row, dict)]

    def place_domestic_cash_order(
        self,
        *,
        ticker: str,
        side: str,
        quantity: int,
        limit_price: float,
        market_code: str = "KRX",
        ord_dvsn: str = "00",
    ) -> dict[str, Any]:
        """Places one domestic cash order and returns API output payload."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if limit_price <= 0:
            raise ValueError("limit_price must be positive")

        side_key = side.strip().lower()
        if side_key == "buy":
            tr_id = "TTTC0012U"
            sll_type = ""
        elif side_key == "sell":
            tr_id = "TTTC0011U"
            sll_type = "01"
        else:
            raise ValueError("side must be buy or sell")

        cano, prod = self._split_account()
        body, _ = self._request(
            method="POST",
            path="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=tr_id,
            params={
                "CANO": cano,
                "ACNT_PRDT_CD": prod,
                "PDNO": ticker.strip().upper(),
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(quantity),
                "ORD_UNPR": f"{limit_price:.0f}",
                "EXCG_ID_DVSN_CD": market_code,
                "SLL_TYPE": sll_type,
                "CNDT_PRIC": "",
            },
        )

        output = body.get("output") or {}
        if isinstance(output, list):
            output = output[0] if output else {}
        return {
            "output": dict(output),
            "msg_cd": str(body.get("msg_cd", "")),
            "msg1": str(body.get("msg1", "")),
        }

    def get_domestic_balance(self, *, inqr_dvsn: str = "02", max_pages: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns domestic account holdings and summary rows."""
        cano, prod = self._split_account()
        rows1: list[dict[str, Any]] = []
        rows2: list[dict[str, Any]] = []

        tr_cont = ""
        fk100 = ""
        nk100 = ""

        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/trading/inquire-balance",
                tr_id="TTTC8434R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "AFHR_FLPR_YN": "N",
                    "OFL_YN": "",
                    "INQR_DVSN": inqr_dvsn,
                    "UNPR_DVSN": "01",
                    "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N",
                    "PRCS_DVSN": "00",
                    "CTX_AREA_FK100": fk100,
                    "CTX_AREA_NK100": nk100,
                },
            )

            output1 = body.get("output1") or []
            output2 = body.get("output2") or []
            if isinstance(output1, dict):
                output1 = [output1]
            if isinstance(output2, dict):
                output2 = [output2]
            rows1.extend(dict(row) for row in output1 if isinstance(row, dict))
            rows2.extend(dict(row) for row in output2 if isinstance(row, dict))

            fk100 = str(body.get("ctx_area_fk100", "") or "")
            nk100 = str(body.get("ctx_area_nk100", "") or "")
            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows1, rows2

    def get_domestic_orderable_cash(self) -> float:
        """Returns the actual orderable cash (주문가능현금) via TTTC8908R."""
        cano, prod = self._split_account()
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id="TTTC8908R",
            params={
                "CANO": cano,
                "ACNT_PRDT_CD": prod,
                "PDNO": "",
                "ORD_UNPR": "0",
                "ORD_DVSN": "00",
                "CMA_EVLU_AMT_ICLD_YN": "Y",
                "OVRS_ICLD_YN": "N",
            },
        )
        output = body.get("output") or {}
        if isinstance(output, list):
            output = output[0] if output else {}

        nrcvb = _to_float(output.get("nrcvb_buy_amt"), default=0.0)
        ord_cash = _to_float(output.get("ord_psbl_cash"), default=0.0)
        ord_sbst = _to_float(output.get("ord_psbl_sbst"), default=0.0)
        ruse = _to_float(output.get("ruse_psbl_amt"), default=0.0)
        logger.info(
            "[cyan]Orderable cash query[/cyan] nrcvb_buy_amt=%.0f ord_psbl_cash=%.0f ord_psbl_sbst=%.0f ruse_psbl_amt=%.0f",
            nrcvb, ord_cash, ord_sbst, ruse,
        )

        # nrcvb_buy_amt = 미수없는매수가능금액 (most conservative)
        cash = nrcvb if nrcvb > 0 else ord_cash
        return max(cash, 0.0)

    def get_overseas_present_balance(
        self,
        *,
        natn_cd: str | None = None,
        tr_mket_cd: str | None = None,
        inqr_dvsn_cd: str = "01",
        wcrc_frcr_dvsn_cd: str = "02",
        max_pages: int = 8,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns paginated overseas account balance snapshots."""
        cano, prod = self._split_account()
        rows1: list[dict[str, Any]] = []
        rows2: list[dict[str, Any]] = []
        rows3: list[dict[str, Any]] = []

        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-stock/v1/trading/inquire-present-balance",
                tr_id="CTRP6504R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "WCRC_FRCR_DVSN_CD": wcrc_frcr_dvsn_cd,
                    "NATN_CD": natn_cd or self.settings.kis_us_natn_cd,
                    "TR_MKET_CD": tr_mket_cd or self.settings.kis_us_tr_mket_cd,
                    "INQR_DVSN_CD": inqr_dvsn_cd,
                },
            )

            output1 = body.get("output1") or []
            output2 = body.get("output2") or []
            output3 = body.get("output3") or []

            if isinstance(output1, dict):
                output1 = [output1]
            if isinstance(output2, dict):
                output2 = [output2]
            if isinstance(output3, dict):
                output3 = [output3]

            rows1.extend(dict(row) for row in output1 if isinstance(row, dict))
            rows2.extend(dict(row) for row in output2 if isinstance(row, dict))
            rows3.extend(dict(row) for row in output3 if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows1, rows2, rows3

    def inquire_overseas_nccs(
        self,
        *,
        exchange_code: str | None = None,
        sort_sqn: str = "DS",
        max_pages: int = 6,
    ) -> list[dict[str, Any]]:
        """Returns overseas unfilled order rows (best-effort pagination)."""
        cano, prod = self._split_account()
        excg = (exchange_code or self.settings.kis_overseas_order_excd).strip().upper()
        fk200 = ""
        nk200 = ""
        tr_cont = ""

        rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-stock/v1/trading/inquire-nccs",
                tr_id="TTTS3018R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "OVRS_EXCG_CD": excg,
                    "SORT_SQN": sort_sqn,
                    "CTX_AREA_FK200": fk200,
                    "CTX_AREA_NK200": nk200,
                },
            )

            output = body.get("output") or []
            if isinstance(output, dict):
                output = [output]
            rows.extend(dict(row) for row in output if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            fk200 = str(body.get("ctx_area_fk200", "") or "")
            nk200 = str(body.get("ctx_area_nk200", "") or "")
            tr_cont = "N"

        return rows

    def inquire_overseas_ccnl(
        self,
        *,
        days: int = 7,
        pdno: str = "",
        exchange_code: str | None = None,
        sort_sqn: str = "DS",
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Returns overseas order/execution rows for recent days (best-effort)."""
        cano, prod = self._split_account()
        excg = (exchange_code or self.settings.kis_overseas_order_excd).strip().upper()

        now = datetime.now(timezone.utc)
        start_dt = (now - timedelta(days=max(days, 1))).strftime("%Y%m%d")
        end_dt = now.strftime("%Y%m%d")

        if not pdno.strip():
            pdno = "" if self.is_demo else "%"

        fk200 = ""
        nk200 = ""
        tr_cont = ""

        rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-stock/v1/trading/inquire-ccnl",
                tr_id="TTTS3035R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "PDNO": pdno.strip().upper(),
                    "ORD_STRT_DT": start_dt,
                    "ORD_END_DT": end_dt,
                    "SLL_BUY_DVSN": "00",
                    "CCLD_NCCS_DVSN": "00",
                    "OVRS_EXCG_CD": excg,
                    "SORT_SQN": sort_sqn,
                    "ORD_DT": "",
                    "ORD_GNO_BRNO": "",
                    "ODNO": "",
                    "CTX_AREA_NK200": nk200,
                    "CTX_AREA_FK200": fk200,
                },
            )

            output = body.get("output") or []
            if isinstance(output, dict):
                output = [output]
            rows.extend(dict(row) for row in output if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            fk200 = str(body.get("ctx_area_fk200", "") or "")
            nk200 = str(body.get("ctx_area_nk200", "") or "")
            tr_cont = "N"

        return rows

    def inquire_overseas_period_trans(
        self,
        *,
        start_date: str,
        end_date: str,
        exchange_code: str | None = None,
        pdno: str = "",
        sll_buy_dvsn_cd: str = "00",
        loan_dvsn_cd: str = "",
        max_pages: int = 8,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns overseas daily transaction rows and summary rows for a date range."""
        cano, prod = self._split_account()
        excg = (exchange_code or self.settings.kis_overseas_order_excd).strip().upper()
        fk100 = ""
        nk100 = ""
        tr_cont = ""

        rows1: list[dict[str, Any]] = []
        rows2: list[dict[str, Any]] = []
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-stock/v1/trading/inquire-period-trans",
                tr_id="CTOS4001R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "ERLM_STRT_DT": str(start_date).strip(),
                    "ERLM_END_DT": str(end_date).strip(),
                    "OVRS_EXCG_CD": excg,
                    "PDNO": pdno.strip().upper(),
                    "SLL_BUY_DVSN_CD": str(sll_buy_dvsn_cd).strip() or "00",
                    "LOAN_DVSN_CD": str(loan_dvsn_cd).strip(),
                    "CTX_AREA_FK100": fk100,
                    "CTX_AREA_NK100": nk100,
                },
            )

            output1 = body.get("output1") or []
            output2 = body.get("output2") or []
            if isinstance(output1, dict):
                output1 = [output1]
            if isinstance(output2, dict):
                output2 = [output2]
            rows1.extend(dict(row) for row in output1 if isinstance(row, dict))
            rows2.extend(dict(row) for row in output2 if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            fk100 = str(body.get("ctx_area_fk100", "") or "")
            nk100 = str(body.get("ctx_area_nk100", "") or "")
            tr_cont = "N"

        return rows1, rows2

    def inquire_domestic_period_profit(
        self,
        *,
        start_date: str,
        end_date: str,
        sort_dvsn: str = "00",
        inqr_dvsn: str = "00",
        cblc_dvsn: str = "00",
        pdno: str = "",
        max_pages: int = 8,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Returns domestic period profit rows and summary rows for a date range."""
        cano, prod = self._split_account()
        fk100 = ""
        nk100 = ""
        tr_cont = ""

        rows1: list[dict[str, Any]] = []
        rows2: list[dict[str, Any]] = []
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/trading/inquire-period-profit",
                tr_id="TTTC8708R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "INQR_STRT_DT": str(start_date).strip(),
                    "INQR_END_DT": str(end_date).strip(),
                    "SORT_DVSN": str(sort_dvsn).strip() or "00",
                    "INQR_DVSN": str(inqr_dvsn).strip() or "00",
                    "CBLC_DVSN": str(cblc_dvsn).strip() or "00",
                    "PDNO": pdno.strip().upper(),
                    "CTX_AREA_FK100": fk100,
                    "CTX_AREA_NK100": nk100,
                },
            )

            output1 = body.get("output1") or []
            output2 = body.get("output2") or []
            if isinstance(output1, dict):
                output1 = [output1]
            if isinstance(output2, dict):
                output2 = [output2]
            rows1.extend(dict(row) for row in output1 if isinstance(row, dict))
            rows2.extend(dict(row) for row in output2 if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            fk100 = str(body.get("ctx_area_fk100", "") or "")
            nk100 = str(body.get("ctx_area_nk100", "") or "")
            tr_cont = "N"

        return rows1, rows2

    def inquire_domestic_daily_ccld(
        self,
        *,
        start_date: str,
        end_date: str,
        pdno: str = "",
        odno: str = "",
        max_pages: int = 8,
    ) -> list[dict[str, Any]]:
        """Returns domestic daily order/execution rows (output1) for date range."""
        cano, prod = self._split_account()
        fk100 = ""
        nk100 = ""
        tr_cont = ""

        rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                tr_id="TTTC0081R",
                tr_cont=tr_cont,
                params={
                    "CANO": cano,
                    "ACNT_PRDT_CD": prod,
                    "INQR_STRT_DT": start_date,
                    "INQR_END_DT": end_date,
                    "SLL_BUY_DVSN_CD": "00",
                    "PDNO": pdno.strip().upper(),
                    "CCLD_DVSN": "00",
                    "INQR_DVSN": "00",
                    "INQR_DVSN_3": "00",
                    "ORD_GNO_BRNO": "",
                    "ODNO": odno.strip(),
                    "INQR_DVSN_1": "",
                    "CTX_AREA_FK100": fk100,
                    "CTX_AREA_NK100": nk100,
                    "EXCG_ID_DVSN_CD": "KRX",
                },
            )

            output1 = body.get("output1") or []
            if isinstance(output1, dict):
                output1 = [output1]
            rows.extend(dict(row) for row in output1 if isinstance(row, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break

            fk100 = str(body.get("ctx_area_fk100", "") or "")
            nk100 = str(body.get("ctx_area_nk100", "") or "")
            tr_cont = "N"

        return rows

    # ── Domestic holiday ──────────────────────────────────────────────
    def get_domestic_holiday(self, bass_dt: str) -> list[dict[str, Any]]:
        """국내휴장일조회 (TR: CTCA0903R).

        Returns rows with ``opnd_yn`` (개장일여부), ``bsns_dy_yn`` (영업일여부),
        ``trad_dy_yn`` (거래일여부) fields.  KIS recommends **1 call/day max**.
        """
        bass_dt = bass_dt.replace("-", "").strip()
        if not bass_dt or len(bass_dt) != 8:
            raise ValueError("bass_dt must be YYYYMMDD")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        fk = ""
        nk = ""
        for _ in range(3):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/quotations/chk-holiday",
                tr_id="CTCA0903R",
                tr_cont=tr_cont,
                params={
                    "BASS_DT": bass_dt,
                    "CTX_AREA_FK": fk,
                    "CTX_AREA_NK": nk,
                },
            )
            page_rows = body.get("output") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(r) for r in page_rows if isinstance(r, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            fk = str(body.get("ctx_area_fk", "") or "")
            nk = str(body.get("ctx_area_nk", "") or "")
            tr_cont = "N"

        return rows

    # ── Domestic investor flow (per-ticker daily) ─────────────────────
    def get_domestic_investor_daily(
        self,
        ticker: str,
        *,
        start_date: str = "",
        end_date: str = "",
        max_pages: int = 4,
    ) -> list[dict[str, Any]]:
        """종목별 투자자 매매동향(일별) (TR: FHPTJ04160001).

        Returns rows with ``frgn_ntby_qty`` (외인순매수), ``orgn_ntby_qty`` (기관순매수).
        """
        from datetime import datetime, timedelta

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")

        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
                tr_id="FHPTJ04160001",
                tr_cont=tr_cont,
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker.strip().upper(),
                    "FID_INPUT_DATE_1": start_date.replace("-", ""),
                    "FID_ORG_ADJ_PRC": "",
                    "FID_ETC_CLS_CODE": "",
                },
            )
            # output2 contains the per-day investor flow data
            page_rows = body.get("output2") or body.get("output1") or body.get("output") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(r) for r in page_rows if isinstance(r, dict))
            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"
        return rows

    # ── Domestic market breadth (advance/decline) ─────────────────────
    def get_domestic_market_breadth(self) -> list[dict[str, Any]]:
        """예상체결 등락 (TR: FHPST01820000). 상승/하락 종목수 포함."""
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/ranking/exp-trans-updown",
            tr_id="FHPST01820000",
            params={
                "fid_rank_sort_cls_code": "0",
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20182",
                "fid_input_iscd": "0000",
                "fid_div_cls_code": "0",
                "fid_aply_rang_prc_1": "",
                "fid_vol_cnt": "",
                "fid_pbmn": "",
                "fid_blng_cls_code": "0",
                "fid_mkop_cls_code": "0",
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(r) for r in rows if isinstance(r, dict)]

    # ── Domestic foreign/institution aggregate ────────────────────────
    def get_domestic_foreign_institution_total(
        self,
        *,
        market_iscd: str = "0001",
        cls_code: str = "0",
    ) -> list[dict[str, Any]]:
        """외국인/기관 순매매 종합 (TR: FHPTJ04400000).

        ``market_iscd``: ``0000`` 전체, ``0001`` 코스피, ``1001`` 코스닥.
        ``cls_code``: ``0`` 전체, ``1`` 외국인, ``2`` 기관계.
        """
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/foreign-institution-total",
            tr_id="FHPTJ04400000",
            params={
                "FID_COND_MRKT_DIV_CODE": "V",
                "FID_COND_SCR_DIV_CODE": "16449",
                "FID_INPUT_ISCD": market_iscd,
                "FID_DIV_CLS_CODE": "0",
                "FID_RANK_SORT_CLS_CODE": "0",
                "FID_ETC_CLS_CODE": cls_code,
                "FID_TRGT_CLS_CODE": "0",
                "FID_TRGT_EXLS_CLS_CODE": "0",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            },
        )
        rows = body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(r) for r in rows if isinstance(r, dict)]

    # ── Domestic earnings estimate ────────────────────────────────────
    def get_domestic_estimate_perform(
        self,
        ticker: str,
    ) -> list[dict[str, Any]]:
        """종목 추정실적 (TR: HHKST668300C0). EPS/매출 컨센서스."""
        body, _ = self._request(
            method="GET",
            path="/uapi/domestic-stock/v1/quotations/estimate-perform",
            tr_id="HHKST668300C0",
            params={
                "SHT_CD": ticker.strip().upper(),
            },
        )
        rows = body.get("output1") or body.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        return [dict(r) for r in rows if isinstance(r, dict)]

    # ── Overseas updown ranking (breadth proxy) ───────────────────────
    def get_overseas_updown_ranking(
        self,
        *,
        excd: str = "NAS",
        nday: str = "0",
        gubn: str = "1",
        max_pages: int = 1,
    ) -> list[dict[str, Any]]:
        """해외주식 상승/하락률 순위 (TR: HHDFS76290000).

        ``gubn``: ``0`` 하락률, ``1`` 상승률.
        ``nday``: ``0`` 당일.
        """
        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/overseas-stock/v1/ranking/updown-rate",
                tr_id="HHDFS76290000",
                tr_cont=tr_cont,
                params={
                    "EXCD": excd.strip().upper(),
                    "NDAY": nday,
                    "GUBN": gubn,
                    "VOL_RANG": "0",
                    "AUTH": "",
                    "KEYB": "",
                },
            )
            page_rows = body.get("output") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(r) for r in page_rows if isinstance(r, dict))
            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"
        return rows

    # ── Domestic dividend schedule (KSD) ──────────────────────────────
    def get_domestic_ksdinfo_dividend(
        self,
        *,
        start_date: str,
        end_date: str,
        sht_cd: str = "",
        gb1: str = "0",
        max_pages: int = 4,
    ) -> list[dict[str, Any]]:
        """예탁원정보 배당일정 (TR: HHKDB669102C0).

        ``gb1``: ``0`` 전체, ``1`` 결산배당, ``2`` 중간배당.
        ``sht_cd``: 종목코드 (빈 문자열이면 전체).
        """
        rows: list[dict[str, Any]] = []
        tr_cont = ""
        for _ in range(max_pages):
            body, headers = self._request(
                method="GET",
                path="/uapi/domestic-stock/v1/ksdinfo/dividend",
                tr_id="HHKDB669102C0",
                tr_cont=tr_cont,
                params={
                    "CTS": "",
                    "GB1": gb1,
                    "F_DT": start_date.replace("-", ""),
                    "T_DT": end_date.replace("-", ""),
                    "SHT_CD": sht_cd.strip(),
                    "HIGH_GB": "",
                },
            )

            page_rows = body.get("output1") or body.get("output") or []
            if isinstance(page_rows, dict):
                page_rows = [page_rows]
            rows.extend(dict(r) for r in page_rows if isinstance(r, dict))

            next_flag = (headers.get("tr_cont") or "").upper()
            if next_flag not in {"M", "F"}:
                break
            tr_cont = "N"

        return rows
