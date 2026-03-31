from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse


def html_response(content: str, *, max_age: int = 30) -> HTMLResponse:
    if max_age <= 0:
        headers = {"Cache-Control": "no-store"}
    else:
        headers = {"Cache-Control": f"public, max-age={max_age}, stale-while-revalidate=60"}
    return HTMLResponse(content, headers=headers)


def json_response(data: Any, *, max_age: int = 30) -> JSONResponse:
    if max_age <= 0:
        headers = {"Cache-Control": "no-store"}
    else:
        headers = {"Cache-Control": f"public, max-age={max_age}, stale-while-revalidate=60"}
    return JSONResponse(content=jsonable_encoder(data), headers=headers)
