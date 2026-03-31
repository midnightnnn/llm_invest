from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml", "jinja2")),
        auto_reload=False,
    )
    env.filters["tojson"] = lambda value: Markup(json.dumps(value, ensure_ascii=False))
    return env


def render_ui_template(template_name: str, /, **context: Any) -> str:
    template = _environment().get_template(template_name)
    return template.render(**context)
