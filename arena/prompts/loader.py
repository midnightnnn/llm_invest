from __future__ import annotations

import functools
from pathlib import Path
from string import Formatter
from typing import Any


def prompt_root() -> Path:
    return Path(__file__).resolve().parent


def prompt_path(*parts: str) -> Path:
    return prompt_root().joinpath(*(str(part).strip("/") for part in parts if str(part).strip("/")))


@functools.lru_cache(maxsize=128)
def load_prompt_text(*parts: str) -> str:
    path = prompt_path(*parts)
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to load prompt template: path={path} err={exc}") from exc


def render_template(template: str, values: dict[str, Any]) -> str:
    safe_values = {key: str(value) for key, value in values.items()}
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(str(template or ""))
        if field_name
    }
    if not fields:
        return str(template or "")
    try:
        return str(template or "").format_map({key: safe_values.get(key, "") for key in fields})
    except Exception:
        return str(template or "")


def render_prompt_text(*parts: str, values: dict[str, Any]) -> str:
    return render_template(load_prompt_text(*parts), values).strip()
