from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from arena.config import load_settings
from arena.ui.app import _build_app
from arena.ui.api_errors import register_api_error_middleware
from tests.test_ui_admin_routes import _DummyRepo


def test_api_error_boundary_logs_traceback_and_returns_request_id(caplog) -> None:
    app = FastAPI()
    register_api_error_middleware(app)

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("db boom")

    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get(
            "/api/boom?tenant_id=tenant-a",
            headers={"X-Request-ID": "req-test-1"},
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
    def blocked():
        raise HTTPException(status_code=403, detail="blocked")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/blocked", headers={"X-Request-ID": "req-403"})

    assert response.status_code == 403
    assert response.headers["X-Request-ID"] == "req-403"
    assert response.json()["detail"] == "blocked"


def test_api_error_boundary_does_not_catch_html_routes(caplog) -> None:
    app = FastAPI()
    register_api_error_middleware(app)

    @app.get("/page")
    def page():
        raise RuntimeError("html boom")

    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get("/page", headers={"X-Request-ID": "req-html"})

    assert response.status_code == 500
    assert not any(getattr(record, "event", "") == "ui_api_request_failed" for record in caplog.records)


def test_built_ui_app_applies_api_error_boundary_to_late_api_routes(caplog) -> None:
    app = _build_app(repo=_DummyRepo(), settings=load_settings())

    @app.get("/api/late-boom")
    def late_boom():
        raise RuntimeError("late api boom")

    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get(
            "/api/late-boom?tenant_id=tenant-b",
            headers={"X-Request-ID": "req-integrated"},
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
