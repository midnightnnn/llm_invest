from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from arena.config import Settings
from arena.market_sources import live_market_sources_for_markets
from arena.models import AccountSnapshot


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {}


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or datetime.now(timezone.utc)).isoformat()


def call_with_optional_tenant(fn, *, tenant_id: str, **kwargs):
    try:
        return fn(**kwargs, tenant_id=tenant_id)
    except TypeError:
        return fn(**kwargs)


def latest_account_snapshot(repo: Any, *, tenant_id: str, market_scope: str | None = None) -> AccountSnapshot | None:
    loader = getattr(repo, "latest_account_snapshot", None)
    if not callable(loader):
        return None
    if str(market_scope or "").strip():
        try:
            return loader(tenant_id=tenant_id, market_scope=market_scope)
        except TypeError:
            pass
    return call_with_optional_tenant(loader, tenant_id=tenant_id)


def repo_metric(repo: Any, method_name: str, default: Any, *, tenant_id: str, **kwargs) -> Any:
    method = getattr(repo, method_name, None)
    if not callable(method):
        return default
    try:
        return method(**kwargs, tenant_id=tenant_id)
    except TypeError:
        try:
            return method(**kwargs)
        except Exception:
            return default
    except Exception:
        return default


def sources_for_settings(settings: Settings) -> list[str] | None:
    if str(getattr(settings, "trading_mode", "") or "").strip().lower() != "live":
        return None
    return live_market_sources_for_markets(settings.kis_target_market) or None


@contextmanager
def repo_tenant_scope(repo: Any, tenant_id: str):
    """Temporarily binds tenant-scoped repository writes for stores lacking tenant args."""
    setter = getattr(repo, "set_tenant_id", None)
    resolver = getattr(repo, "resolve_tenant_id", None)
    previous = ""
    if callable(resolver):
        try:
            previous = str(resolver() or "").strip().lower()
        except Exception:
            previous = ""
    elif hasattr(repo, "tenant_id"):
        previous = str(getattr(repo, "tenant_id", "") or "").strip().lower()

    if callable(setter):
        setter(tenant_id)
    elif hasattr(repo, "tenant_id"):
        try:
            setattr(repo, "tenant_id", tenant_id)
        except Exception:
            pass
    try:
        yield
    finally:
        if previous:
            if callable(setter):
                setter(previous)
            elif hasattr(repo, "tenant_id"):
                try:
                    setattr(repo, "tenant_id", previous)
                except Exception:
                    pass


def snapshot_payload(snapshot: AccountSnapshot, *, tenant_id: str, max_positions: int) -> dict[str, Any]:
    raw = model_dump(snapshot)
    positions = raw.get("positions") if isinstance(raw.get("positions"), dict) else {}
    ordered_positions = sorted(
        positions.values(),
        key=lambda row: safe_float((row or {}).get("quantity")) * safe_float((row or {}).get("market_price_krw")),
        reverse=True,
    )
    if max_positions > 0:
        ordered_positions = ordered_positions[: max(1, min(int(max_positions), 100))]
    return {
        "status": "ok",
        "tenant_id": tenant_id,
        "cash_krw": safe_float(raw.get("cash_krw")),
        "total_equity_krw": safe_float(raw.get("total_equity_krw")),
        "usd_krw_rate": safe_float(raw.get("usd_krw_rate")),
        "cash_foreign": safe_float(raw.get("cash_foreign")),
        "cash_foreign_currency": str(raw.get("cash_foreign_currency") or ""),
        "position_count": len(positions),
        "positions": ordered_positions,
    }
