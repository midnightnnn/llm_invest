from __future__ import annotations

import asyncio
import inspect
import json
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class DirectResponse:
    def __init__(self, response: Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return int(getattr(self._response, "status_code", 200))

    @property
    def headers(self):
        return getattr(self._response, "headers", {})

    @property
    def body(self) -> bytes:
        return bytes(getattr(self._response, "body", b""))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self):
        return json.loads(self.text or "{}")


class DirectRouteClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.session: dict[str, object] = {}

    def close(self) -> None:
        return None

    def get(self, path: str, *, params: dict | None = None, follow_redirects: bool = True) -> DirectResponse:
        _ = follow_redirects
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        data: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
        follow_redirects: bool = True,
    ) -> DirectResponse:
        _ = follow_redirects
        return self._request("POST", path, data=data, json=json, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
    ) -> DirectResponse:
        endpoint, path_params = self._resolve_endpoint(method, path)
        query_items = list(parse_qsl(urlsplit(path).query, keep_blank_values=True))
        if params:
            query_items.extend((str(k), str(v)) for k, v in params.items())
        query_string = urlencode(query_items, doseq=True).encode("utf-8")
        body = b""
        headers: list[tuple[bytes, bytes]] = []
        if json is not None:
            body = __import__("json").dumps(json).encode("utf-8")
            headers.append((b"content-type", b"application/json"))

        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        req = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": method.upper(),
                "scheme": "http",
                "path": urlsplit(path).path,
                "raw_path": urlsplit(path).path.encode("utf-8"),
                "query_string": query_string,
                "headers": headers,
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "app": self.app,
                "session": self.session,
            },
            receive=receive,
        )
        values = {}
        merged: dict[str, object] = dict(path_params)
        for qk, qv in query_items:
            merged[qk] = qv
        if params:
            merged.update(params)
        if data:
            merged.update(data)
        for name in inspect.signature(endpoint).parameters:
            if name == "request":
                continue
            if name in merged:
                values[name] = merged[name]
                continue
            default = inspect.signature(endpoint).parameters[name].default
            if default is inspect.Signature.empty:
                continue
            resolved = getattr(default, "default", default)
            if resolved is inspect.Signature.empty:
                continue
            values[name] = resolved
        result = endpoint(req, **values)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        if not isinstance(result, Response):
            result = JSONResponse(result)
        return DirectResponse(result)

    def _resolve_endpoint(self, method: str, path: str):
        request_path = urlsplit(path).path
        method_token = str(method or "").strip().upper()
        for route in self.app.routes:
            route_path = getattr(route, "path", "")
            route_methods = getattr(route, "methods", set()) or set()
            if route_path == request_path and method_token in route_methods:
                return route.endpoint, {}
            path_regex = getattr(route, "path_regex", None)
            if path_regex is None or method_token not in route_methods:
                continue
            match = path_regex.match(request_path)
            if not match:
                continue
            path_params = match.groupdict()
            for key, value in list(path_params.items()):
                convertor = getattr(route, "param_convertors", {}).get(key)
                if convertor is not None:
                    path_params[key] = convertor.convert(value)
            return route.endpoint, path_params
        raise LookupError(f"route not found: {method_token} {request_path}")
