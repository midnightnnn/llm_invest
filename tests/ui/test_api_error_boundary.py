from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException
import httpx

from arena.config import load_settings
from arena.ui.app import _build_app
from arena.ui.api_errors import register_api_error_middleware
from tests.ui.helpers import _DummyRepo


async def _asgi_get(app: FastAPI, path: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path, headers=headers)


def test_api_error_boundary_logs_traceback_and_returns_request_id(caplog) -> None:
    app = FastAPI()
    register_api_error_middleware(app)

    @app.get("/api/boom")
    async def boom():
        raise RuntimeError("db boom")

    with caplog.at_level(logging.ERROR):
        response = asyncio.run(
            _asgi_get(
                app,
                "/api/boom?tenant_id=tenant-a",
                headers={"X-Request-ID": "req-test-1"},
            )
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-test-1"
    assert response.json() == {
        "status": "error",
        "error": "internal_server_error",
        "request_id": "req-test-1",
    }
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "ui_api_request_failed"
    )
    assert failure_record.exc_info is not None
    assert getattr(failure_record, "request_id", "") == "req-test-1"
    assert getattr(failure_record, "tenant_id", "") == "tenant-a"
    assert getattr(failure_record, "path", "") == "/api/boom"
    assert getattr(failure_record, "method", "") == "GET"


def test_api_error_boundary_preserves_http_errors() -> None:
    app = FastAPI()
    register_api_error_middleware(app)

    @app.get("/api/blocked")
    async def blocked():
        raise HTTPException(status_code=403, detail="blocked")

    response = asyncio.run(_asgi_get(app, "/api/blocked", headers={"X-Request-ID": "req-403"}))

    assert response.status_code == 403
    assert response.headers["X-Request-ID"] == "req-403"
    assert response.json()["detail"] == "blocked"


def test_api_error_boundary_does_not_catch_html_routes(caplog) -> None:
    app = FastAPI()
    register_api_error_middleware(app)

    @app.get("/page")
    async def page():
        raise RuntimeError("html boom")

    with caplog.at_level(logging.ERROR):
        response = asyncio.run(_asgi_get(app, "/page", headers={"X-Request-ID": "req-html"}))

    assert response.status_code == 500
    assert not any(getattr(record, "event", "") == "ui_api_request_failed" for record in caplog.records)


def test_built_ui_app_applies_api_error_boundary_to_late_api_routes(caplog) -> None:
    app = _build_app(repo=_DummyRepo(), settings=load_settings())

    @app.get("/api/late-boom")
    async def late_boom():
        raise RuntimeError("late api boom")

    with caplog.at_level(logging.ERROR):
        response = asyncio.run(
            _asgi_get(
                app,
                "/api/late-boom?tenant_id=tenant-b",
                headers={"X-Request-ID": "req-integrated"},
            )
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-integrated"
    assert response.json()["request_id"] == "req-integrated"
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "ui_api_request_failed"
    )
    assert failure_record.exc_info is not None
    assert getattr(failure_record, "tenant_id", "") == "tenant-b"
