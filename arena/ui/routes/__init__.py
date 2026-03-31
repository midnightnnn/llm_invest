from __future__ import annotations

from arena.ui.routes.auth import register_auth_routes
from arena.ui.routes.ops import register_ops_routes
from arena.ui.routes.sleeves import register_sleeve_routes

__all__ = ["register_auth_routes", "register_ops_routes", "register_sleeve_routes"]
